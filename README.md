# Fine-tuned-Small-Language-Models-for-Summarization

## Features
- Computational Resource Optimization (LoRA + 4-bit quantization)
- Efficient Text Summarization Using SLM
![image](https://github.com/user-attachments/assets/5556c880-7d22-47a4-8e3a-894ee8d9196d)

## Environment
- Python 3.9.19
- Pytorch 2.6.0
- CUDA 11.8

## Installation
```
git clone https://github.com/fangevo/Fine-tuned-Small-Language-Model-for-Summarization.git
```
## Usage
1. 
```
cd Fine-tuned-Small-Language-Model-for-Summarization
```
2. Run create_dataset.py to generate reference summaries using Qwen2.5-14B.
```
Python create_dataset.py
```
3. Run fine_tuning.py to fine-tune the small language model using the reference summary.
```
Python fine_tuning.py
```
## Fine-tuning results
![image](https://github.com/user-attachments/assets/ba884c65-4375-4d05-a5d8-90ca056535ba)
