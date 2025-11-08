import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model paths
base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "/scratch/sujayb/anlp/clarifying_questions/data/llama2/gen_clarify_q/DPO_1/best_checkpoint"
output = "/scratch/sujayb/anlp/clarifying_questions/DPO_1_merged"

# Create output directory
os.makedirs(output, exist_ok=True)

# Save the adapter config for reference
adapter_config_path = os.path.join(adapter, "adapter_config.json")
if os.path.exists(adapter_config_path):
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    with open(os.path.join(output, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

# Save the training config for reference
training_config_path = os.path.join(os.path.dirname(adapter), "config.json")
if os.path.exists(training_config_path):
    with open(training_config_path, "r") as f:
        training_config = json.load(f)
    with open(os.path.join(output, "config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

# Load base model
print(f"Loading base model from {base}...")
model = AutoModelForCausalLM.from_pretrained(
    base,
    load_in_8bit=True,
    device_map={'': 0}  # Assuming single GPU
)

# Load and merge LoRA adapter
print(f"Loading adapter from {adapter}...")
model = PeftModel.from_pretrained(
    model,
    adapter,
    is_trainable=False,
)

# Merge weights and unload original adapter
print("Merging adapter weights...")
model = model.merge_and_unload()

# Save the merged model
print(f"Saving merged model to {output}...")
model.save_pretrained(output)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.save_pretrained(output)

print("Done!")
