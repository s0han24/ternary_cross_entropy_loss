import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "./final_fix/final_model_merged"

# Load tokenizer
# use only gpu 0
tokenizer = AutoTokenizer.from_pretrained(adapter_path, device_map={"": 0})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    adapter_path,
    device_map={"": 0}
)
model.eval()
print("Base model loaded.")

def generate_clarification_batch(questions):
    batch_messages = []
    for question in questions:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates clarifying questions to gain insights on the user's intent."},
            {"role": "user", "content": question},
        ]
        batch_messages.append(messages)
    
    # Apply chat template to all messages
    inputs = tokenizer.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    results = []
    for output in outputs:
        results.append(tokenizer.decode(output, skip_special_tokens=True))
    
    return results



# Read prompts from JSONL file
def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list

# Main inference loop
jsonl_file = "./data/ambigqa.dev_4h.clarify.jsonl"  # Change this to your JSONL file path
batch_size = 32

data_list = read_jsonl(jsonl_file)
print(f"Loaded {len(data_list)} examples from {jsonl_file}")

# Extract questions for inference
prompts = [item['question'] for item in data_list]

all_results = []

# Process in batches
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
    
    results = generate_clarification_batch(batch)
    all_results.extend(results)

# Save results to JSONL with id, question, and clarification_question fields
with open("cq_sft_dev_100.jsonl", 'w', encoding='utf-8') as f:
    for data_item, result in zip(data_list, all_results):
        output = {
            "id": data_item['id'],
            "question": data_item['question'],
            "clarification_question": result
        }
        f.write(json.dumps(output) + '\n')

print("\nResults saved to results.jsonl")

# Print first few examples
print("\n--- Sample Results ---")
for idx in range(min(5, len(data_list))):
    print(f"\nExample {idx + 1}:")
    print(f"ID: {data_list[idx]['id']}")
    print(f"Question: {data_list[idx]['question']}")
    print(f"Clarification: {all_results[idx]}")