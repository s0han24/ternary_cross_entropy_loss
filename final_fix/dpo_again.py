import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import json

# ----------------------------
# Paths
# ----------------------------
train_path = "./splits/final_train.jsonl"
dev_path = "./splits/final_dev.jsonl"
base_model = "meta-llama/Llama-3.2-3B-Instruct"

# ----------------------------
# Load data
# ----------------------------
with open(train_path, 'r') as f:
    train_data = [json.loads(l) for l in f]
with open(dev_path, 'r') as f:
    dev_data = [json.loads(l) for l in f]

# ----------------------------
# Convert dataset → DPO format
# Each example must produce: prompt, chosen, rejected
# ----------------------------
def format_dpo(ex, rank_threshold=1.5):
    clarifications = sorted(ex['dpo']['clarifications'], key=lambda x: x['rank'])

    # Chosen = best clarification
    chosen = clarifications[0]['question']

    # Rejected = worst clarification >= rank_threshold
    rejected_list = [c['question'] for c in clarifications if c['rank'] >= rank_threshold]
    if len(rejected_list) == 0:
        rejected = clarifications[-1]['question']   # fallback → worst
    else:
        rejected = rejected_list[-1]

    return {
        'prompt': ex['question'],
        'chosen': chosen + "\n",
        'rejected': rejected + "\n"
    }

# ----------------------------
# Construct flattened DPO datasets
# ----------------------------
train_flat = [format_dpo(ex) for ex in train_data]  # subset
dev_flat = [format_dpo(ex) for ex in dev_data]

train_dataset = Dataset.from_dict({
    'prompt': [ex['prompt'] for ex in train_flat],
    'chosen': [ex['chosen'] for ex in train_flat],
    'rejected': [ex['rejected'] for ex in train_flat]
})

dev_dataset = Dataset.from_dict({
    'prompt': [ex['prompt'] for ex in dev_flat],
    'chosen': [ex['chosen'] for ex in dev_flat],
    'rejected': [ex['rejected'] for ex in dev_flat]
})

# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ----------------------------
# Load base model
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    device_map="auto"
)

# ----------------------------
# LoRA config
# ----------------------------
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ----------------------------
# DPO Training Config
# ----------------------------
training_args = DPOConfig(
    output_dir="./dpo_results_1",
    num_train_epochs=1,
    per_device_train_batch_size=8,     # DPO is heavier, use lower batch size
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,                # DPO requires lower LR
    beta=0.1,                          # Standard DPO beta
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    max_grad_norm=0.2,
    ddp_find_unused_parameters=False,
)

# ----------------------------
# Trainer
# ----------------------------
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.take(32),
    eval_dataset=dev_dataset.take(10),
    processing_class=tokenizer,         # DPOTrainer uses tokenizer instead of processing_class
)

trainer.train()

# ----------------------------
# Save full model
# ----------------------------
print("Saving fine-tuned DPO model...")
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./final_dpo_model_merged")
tokenizer.save_pretrained("./final_dpo_model_merged")
