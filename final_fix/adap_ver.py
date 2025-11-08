from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

base_model = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model WITHOUT adapter...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    dtype=torch.bfloat16,  # Use bfloat16 for base model test
    low_cpu_mem_usage=True
)

model.eval()

prompt = "What is the capital of France?"
messages = [{"role": "user", "content": prompt}]

formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(formatted_prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("\nGenerating with base model (no adapter)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nBase Model Response:")
print(generated_text)