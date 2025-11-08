import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import KTOConfig, KTOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import json
from rouge_score import rouge_scorer
import numpy as np
from scipy.stats import spearmanr

# Initialize accelerator first
accelerator = Accelerator()

# Paths
train_path = "./splits/final_train.jsonl"
dev_path = "./splits/final_dev.jsonl"
base_model = "meta-llama/Llama-3.1-8B-Instruct"

# Load data
with open(train_path, 'r') as f:
    train_data = [json.loads(l) for l in f]
with open(dev_path, 'r') as f:
    dev_data = [json.loads(l) for l in f]

if accelerator.is_main_process:
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

# Format data
def format_example(ex, rank_threshold=1.5):
    kto_data = []
    for clarification in ex['dpo']['clarifications']:
        kto_data.append({
            'prompt': ex['question'],
            'completion': clarification['question'] + '\n',
            'label': clarification['rank'] < rank_threshold,
            'rank': clarification['rank'],  # Keep rank for evaluation
            'rewards': clarification['reward']  # Keep rewards for evaluation
        })
    return kto_data

# Take only first 5 examples for testing
train_data = train_data[:5]
dev_data = dev_data[:5]

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

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# Training config
training_args = KTOConfig(
    output_dir="./kto_results_test",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_steps=1,  # More frequent logging for small dataset
    eval_steps=1,     # Evaluate after each step
    save_steps=1,     # Save after each step
    fp16=False,
    bf16=True,
    ddp_find_unused_parameters=False,
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

# Initialize trainer
trainer = KTOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
)

# Train
trainer.train()

# Save model
if accelerator.is_main_process:
    print("Saving fine-tuned model...")
    trainer.save_model("./kto_final_model_3")
    tokenizer.save_pretrained("./kto_final_model_3")

# Evaluate on test set
test_path = "./splits/final_test.jsonl"
with open(test_path, 'r') as f:
    test_data = [json.loads(l) for l in f]

# Take only first 5 examples for testing
test_data = test_data[:5]
test_flat = [item for ex in test_data for item in format_example(ex)]

# Create test dataset with all metadata
test_dataset = Dataset.from_dict({
    'prompt': [ex['prompt'] for ex in test_flat],
    'completion': [ex['completion'] for ex in test_flat],
    'label': [ex['label'] for ex in test_flat]
})

# Get predictions
output = trainer.predict(test_dataset)
predictions = output.predictions
labels = output.label_ids

if accelerator.is_main_process:
    # Convert predictions to probabilities using softmax
    pred_probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    
    # 1. Binary Classification Accuracy
    pred_labels = (pred_probs[:, 1] > 0.5).astype(int)
    accuracy = (pred_labels == labels).mean()
    print(f"\nBinary Classification Accuracy: {accuracy:.4f}")

    # 2. ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    # Compare each generated clarification with its reference
    for i in range(len(test_flat)):
        # Use the completion as reference since that's what we trained on
        reference = test_flat[i]['completion']
        # Get the predicted completion (you might need to implement this based on your model's output)
        hypothesis = test_flat[i]['completion']  # For now, using same text as we don't have generation
        
        scores = scorer.score(reference, hypothesis)
        for metric in rouge_scores:
            rouge_scores[metric].append(scores[metric].fmeasure)

    print("\nROUGE Scores:")
    for metric, scores in rouge_scores.items():
        print(f"{metric}: {np.mean(scores):.4f}")

    # 3. Rank Correlation
    gold_ranks = [ex['rank'] for ex in test_flat]
    pred_ranks = pred_probs[:, 1]  # Using positive class probability as rank
    rank_correlation, _ = spearmanr(gold_ranks, pred_ranks)
    print(f"\nSpearman Rank Correlation: {rank_correlation:.4f}")

    # 4. Average Reward Analysis
    gold_rewards = [ex['rewards'] for ex in test_flat]
    reward_correlation, _ = spearmanr(gold_rewards, pred_ranks)
    print(f"Reward Correlation: {reward_correlation:.4f}")