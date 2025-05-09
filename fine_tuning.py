import os
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import bitsandbytes as bnb
from evaluate import load
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np
from modelscope import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# Read JSON data
df = pd.read_json("data/summarization_dataset.json", lines=False)

# Divide training, validation, and test sets（80%/10%/10%）
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_json("train.json", orient="records", lines=True, force_ascii=False)
val_df.to_json("validation.json", orient="records", lines=True, force_ascii=False)
test_df.to_json("test.json", orient="records", lines=True, force_ascii=False)

dataset = load_dataset("json", data_files={"train": "train.json", "validation": "validation.json", "test": "test.json"})




# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

# Construct prompt
def build_chat_prompt(document):
    messages = [
        {"role": "system", "content": "你是一个擅长做文本摘要的助手。你的任务是为用户提供精准的新闻摘要，不要扩展内容或提供个人意见。"},
        {"role": "user", "content": f"请为以下新闻内容生成简洁凝练的摘要：\n\n{document}"}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def preprocess_function(examples):

    inputs = [build_chat_prompt(doc) for doc in examples["document"]]
    summaries = examples["summary"]

    input_encodings = tokenizer(inputs, truncation=True, max_length=1024, padding=False)
    summary_encodings = tokenizer(summaries, truncation=True, max_length=256, padding=False)

    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(inputs)):
        prompt_ids = input_encodings["input_ids"][i]
        prompt_attention = input_encodings["attention_mask"][i]
        summary_ids = summary_encodings["input_ids"][i]

        labels = [-100] * len(prompt_ids) + summary_ids

        combined_input_ids = prompt_ids + summary_ids
        combined_attention_mask = prompt_attention + [1] * len(summary_ids)

        model_inputs["input_ids"].append(combined_input_ids)
        model_inputs["attention_mask"].append(combined_attention_mask)
        model_inputs["labels"].append(labels)

    return model_inputs

# map preprocessing
dataset = dataset.map(preprocess_function, batched=True, batch_size=8)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]




# bnb 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model_id = "qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True, device_map="auto")

# Lora
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj","o_proj"],
)

model = get_peft_model(base_model, lora_config)
model.to("cuda")

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.4f}%")

print_trainable_parameters(model)
model.config.pad_token_id = tokenizer.pad_token_id

training_args = TrainingArguments(
    output_dir="experiments",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    warmup_steps=500,
    bf16=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="tensorboard",
    evaluation_strategy="epoch",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# checkpoint
checkpoint_path = None
if os.path.isdir(training_args.output_dir):
    checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Select the latest checkpoint
        checkpoint_path = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"Breakpoint detected, ready to resume training from {checkpoint_path}")

model.config.use_cache = False
trainer.train(resume_from_checkpoint=checkpoint_path)

# valuation
model.eval()
rouge = load("rouge")
bleurt = load("bleurt")

def generate_summary(text, use_beam_search=False, num_beams=4):
    prompt = build_chat_prompt(text)
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
    input_length = tokenized["input_ids"].shape[1]
    max_new_tokens = int(input_length * 0.3)
    max_new_tokens = max(20, min(max_new_tokens, 200))

    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda", non_blocking=True)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": False,
        "pad_token_id": tokenizer.pad_token_id
    }

    if use_beam_search:
        generation_args["num_beams"] = num_beams
        generation_args["early_stopping"] = True

    with torch.no_grad():
        summary_ids = model.generate(**inputs, **generation_args)

    generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    del inputs, summary_ids, tokenized
    torch.cuda.empty_cache()


    if "assistant\n" in generated_text:
        return generated_text.split("assistant\n")[-1].strip()
    return generated_text.strip()


# Generate a summary of the test set text
test_texts = test_df["document"].tolist()
reference_summaries = test_df["summary"].tolist()
generated_summaries = []
print("\nStart generating summaries and evaluating:")
for idx, text in enumerate(tqdm(test_texts, desc="Generating summary progress")):
    summary = generate_summary(text, use_beam_search=True, num_beams=5)
    generated_summaries.append(summary)

#Rouge Score
rouge_scores = rouge.compute(predictions=generated_summaries, references=reference_summaries)
print("ROUGE Scores:", rouge_scores)

# BLEURT Score
bleurt_scores_list = []
for pred, ref in tqdm(zip(generated_summaries, reference_summaries), total=len(generated_summaries), desc="BLEURT Calculating progress"):
    bleurt_score = bleurt.compute(predictions=[pred], references=[ref])['scores'][0]
    bleurt_scores_list.append(bleurt_score)

average_bleurt = np.mean(bleurt_scores_list)
print(f"\nAverage BLEURT Score: {average_bleurt:.4f}")

num_examples_to_show = 5
for i in range(num_examples_to_show):
    print(f"\n {i+1}:")
    print(f"Original text:\n{test_texts[i]}")
    print(f"Reference summary:\n{reference_summaries[i]}")
    print(f"Generating Summaries:\n{generated_summaries[i]}")
    print("=" * 80)
