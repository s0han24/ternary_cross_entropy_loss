import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_path = "meta-llama/Llama-3.2-3B-Instruct"
# adapter_path = "./final_fix/final_kto_model_merged"
jsonl_file = "./ca_base_dev_200_purified.jsonl"
output_file = "fa_base_dev_200_base.jsonl"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path, device_map={"": 0})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    adapter_path,
    device_map={"": 0}
)
model.eval()
print("Model loaded.")

def generate_final_answer_batch(questions, clarifying_questions, clarifying_answers):
    batch_messages = []
    for question, cq, ca in zip(questions, clarifying_questions, clarifying_answers):
        # Construct the conversation with clarification context
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the clarification provided by the user."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": cq},
            {"role": "user", "content": ca},
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
            max_new_tokens=128,  # Increased for full answers
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
batch_size = 32

data_list = read_jsonl(jsonl_file)
print(f"Loaded {len(data_list)} examples from {jsonl_file}")

# Extract data for inference
questions = [item['question'] for item in data_list]
clarifying_questions = [item['clarification_question'] for item in data_list]
clarifying_answers = [item['clarification_answer'] for item in data_list]

all_results = []

# Process in batches
for i in range(0, len(questions), batch_size):
    batch_q = questions[i:i + batch_size]
    batch_cq = clarifying_questions[i:i + batch_size]
    batch_ca = clarifying_answers[i:i + batch_size]
    
    print(f"Processing batch {i//batch_size + 1}/{(len(questions)-1)//batch_size + 1}")
    
    results = generate_final_answer_batch(batch_q, batch_cq, batch_ca)
    all_results.extend(results)



# Save results to JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for data_item, result in zip(data_list, all_results):
        output = {
            "id": data_item['id'],
            "question": data_item['question'],
            "clarification_question": data_item['clarification_question'],
            "clarification_answer": data_item['clarification_answer'],
            "final_answer": result.split('assistant\n\n')[-1].strip(),
            "ground_truth_answer": data_item.get('ground_truth_answer', '')
        }
        f.write(json.dumps(output) + '\n')

print(f"\nResults saved to {output_file}")

# Print first few examples
print("\n--- Sample Results ---")
for idx in range(min(5, len(data_list))):
    print(f"\nExample {idx + 1}:")
    print(f"ID: {data_list[idx]['id']}")
    print(f"Question: {data_list[idx]['question']}")
    print(f"Clarification Answer: {data_list[idx]['clarification_answer']}")
    print(f"Final Answer: {all_results[idx]}")
    print(f"Ground Truth: {data_list[idx].get('ground_truth_answer', 'N/A')}")