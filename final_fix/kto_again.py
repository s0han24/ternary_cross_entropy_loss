import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import KTOConfig, KTOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import json

# Initialize accelerator first
# Paths
train_path = "./splits/final_train.jsonl"
dev_path = "./splits/final_dev.jsonl"
base_model = "meta-llama/Llama-3.2-3B-Instruct"

# Load data
with open(train_path, 'r') as f:
    train_data = [json.loads(l) for l in f]
with open(dev_path, 'r') as f:
    dev_data = [json.loads(l) for l in f]

# Format data
def format_example(ex, rank_threshold=1.5):
    kto_data = []
    for clarification in ex['dpo']['clarifications']:
        kto_data.append({
            'prompt': ex['question'],
            'completion': clarification['question'] + '\n',
            'label': clarification['rank'] < rank_threshold,
        })
    return kto_data

# Flatten datasets
train_flat = [item for ex in train_data for item in format_example(ex)]
dev_flat = [item for ex in dev_data for item in format_example(ex)]

train_dataset = Dataset.from_dict({
    'prompt': [ex['prompt'] for ex in train_flat],
    'completion': [ex['completion'] for ex in train_flat],
    'label': [ex['label'] for ex in train_flat]
})

dev_dataset = Dataset.from_dict({
    'prompt': [ex['prompt'] for ex in dev_flat],
    'completion': [ex['completion'] for ex in dev_flat],
    'label': [ex['label'] for ex in dev_flat]
})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model without device_map for distributed training
# Remove quantization_config if you want full precision training
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.float32,
)

# Training config
training_args = KTOConfig(
    output_dir="./kto_results_1",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,  # Lower LR for full fine-tuning
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    ddp_find_unused_parameters=False,
    max_grad_norm=0.2
)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Get PEFT model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Print number of trainable parameters

# Initialize trainer with PEFT model
trainer = KTOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
)

trainer.train()

# Save full model (only on main process)
print("Saving fine-tuned model...")
trained_model = trainer.model.merge_and_unload()
trained_model.save_pretrained("./final_kto_model_merged")
tokenizer.save_pretrained("./final_kto_model_merged")

# # Run inferences on the test set
# test_path = "./splits/final_test.jsonl"
# with open(test_path, 'r') as f:
#     test_data = [json.loads(l) for l in f]
# test_flat = [item for ex in test_data for item in format_example(ex)]
# test_dataset = Dataset.from_dict({
#     'prompt': [ex['prompt'] for ex in test_flat],
#     'completion': [ex['completion'] for ex in test_flat],
#     'label': [ex['label'] for ex in test_flat]
# })
# preds = trainer.predict(test_dataset)
# # Simple accuracy
# pred_labels = (preds.predictions[:, 1] > preds.predictions[:, 0]).astype(int)
# accuracy = (pred_labels == preds.label_ids).mean()
# if accelerator.is_main_process:
#     print(f"Test Accuracy: {accuracy:.4f}")


############################
# RUN ON SFT OUTPUT MODEL
# Run on full dataset
# Run DPO also on SFT output model
# Run test
############################