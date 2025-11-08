import torch
from accelerate import Accelerator
accelerator = Accelerator()
device_map = {"": accelerator.device}

import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

train_path = "./splits/final_train.jsonl"
dev_path = "./splits/final_dev.jsonl"
test_path = "./splits/final_test.jsonl"
base_model = "meta-llama/Llama-3.2-3B-Instruct"

# Load data
with open(train_path, 'r') as f:
    train_data = [json.loads(l) for l in f]
with open(dev_path, 'r') as f:
    dev_data = [json.loads(l) for l in f]
with open(test_path, 'r') as f:
    test_data = [json.loads(l) for l in f]

print(f"Train data size: {len(train_data)}")
print(f"Dev data size: {len(dev_data)}")
print(f"Test data size: {len(test_data)}")

# Format data for SFT - combine input and output into single text
def format_example(example):
    return f"Question: {example['question']}\nClarifying Question: {example['clarification']['question']}"

train_texts = [format_example(ex) for ex in train_data]
dev_texts = [format_example(ex) for ex in dev_data]
test_texts = [format_example(ex) for ex in test_data]

train_dataset = Dataset.from_dict({'text': train_texts})
dev_dataset = Dataset.from_dict({'text': dev_texts})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model *before* SFTTrainer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.float32,
    device_map=accelerator.device,   # IMPORTANT
)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Configure training
sft_config = SFTConfig(
    output_dir="./results_1",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    ddp_find_unused_parameters=False,  # Add this
    max_grad_norm=0.2
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()

# merge and save the model
trained_model = trainer.model.merge_and_unload()
trained_model.save_pretrained("./final_model_merged")
tokenizer.save_pretrained("./final_model_merged")