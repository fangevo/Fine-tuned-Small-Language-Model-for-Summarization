import os
import torch
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import jieba
from unidecode import unidecode
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Loading Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Qwen/Qwen2.5-14B-Instruct
model_name = "Qwen/Qwen2.5-14B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Use of double quantization
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Preprocessed text
def clean_news_text(text):
  
    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：“”‘’（）《》%]", "", text)
    text = text.replace("《", '"').replace("》", '"')
    text = re.sub(r"(\d+)%", r"\1 %", text)
    text = " ".join(jieba.cut(text))
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Loading CSV data
csv_file = "data/processed_news.csv"  
df = pd.read_csv(csv_file)

if "content" not in df.columns:
    raise ValueError("CSV file is missing the 'content' column.")

df["content"] = df["content"].fillna("").apply(clean_news_text)
df = df.dropna(subset=["content"]).drop_duplicates(subset=["content"])

clean_csv_file = "cleaned_news_data.csv"
df.to_csv(clean_csv_file, index=False, encoding="utf-8")

print(f"Cleaned news dataset saved: {clean_csv_file}")



# Generating Summaries
def adjust_max_new_tokens(text, min_tokens=100, max_tokens=300, ratio=0.3):
    
    return max(min_tokens, min(max_tokens, int(len(text) * ratio)))


def generate_summaries(texts, batch_size):

    messages_list = [
        [
            {"role": "system", "content": "你是一个擅长做文本摘要的助手。你的任务是为用户提供精准的新闻摘要，不要扩展内容或提供个人意见。"},
            {"role": "user", "content": f"请为以下新闻内容生成简洁凝练的摘要：\n\n{text}"}
        ]
        for text in texts
    ]

    formatted_texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
    tokenizer.padding_side = "left"

    inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True).to("cuda", non_blocking=True)
    model.to(device)

    max_tokens_list = [adjust_max_new_tokens(text) for text in texts]

    output_ids = [
        model.generate(**{k: v[i].unsqueeze(0) for k, v in inputs.items()}, 
                       max_new_tokens=max_tokens_list[i], 
                       do_sample=True, 
                       temperature=0.7, 
                       use_cache=False)[0]
        for i in range(len(texts))
    ]

    summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    summaries = [summary.split("assistant\n")[-1].strip() for summary in summaries]

    return summaries



# Checkpoint 
output_file = "summarization_dataset.json"

if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        try:
            dataset = json.load(f)
            processed_texts = {item["document"] for item in dataset if item.get("summary")}
        except json.JSONDecodeError:
            print("Error: JSON broken")
            dataset = []
            processed_texts = set()
else:
    dataset = []
    processed_texts = set()

csv_file = "cleaned_news_data.csv"
df = pd.read_csv(csv_file)

if "content" not in df.columns:
    raise ValueError("CSV file missing 'content' column")

documents = df["content"].dropna().drop_duplicates().tolist()

pending_docs = [doc for doc in documents if doc not in processed_texts]

if not pending_docs:
    print("All news summaries have been completed")
else:
    print(f"Found  {len(pending_docs)} news items that did not generate a summary，start generating...")

batch_size = 64
save_interval = 10

for i in tqdm(range(0, len(pending_docs), batch_size), desc="Processing"):
    batch_docs = pending_docs[i:i + batch_size]

    try:
        batch_summaries = generate_summaries(batch_docs, batch_size)
        new_data = [{"document": doc, "summary": summary} for doc, summary in zip(batch_docs, batch_summaries)]
        dataset.extend(new_data)

        if len(dataset) % save_interval < batch_size:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            print(f"Save {len(dataset)}")

    except Exception as e:
        dataset.extend([{"document": doc, "summary": "ERROR"} for doc in batch_docs])

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"News summary dataset saved：{output_file}")
print("Dataset split completed.")

