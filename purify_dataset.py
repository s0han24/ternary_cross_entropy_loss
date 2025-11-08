import json


dataset_path = "./ca_base_dev_200.jsonl"

# read the json files
def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list

# Main processing
print(f"Reading from {dataset_path}...")
data_list = read_jsonl(dataset_path)

# for each attribute, split at "assistant\n\n" and only keep the last part
# this is to remove any previous model outputs

for item in data_list:
    if 'clarification_question' in item:
        parts = item['clarification_question'].split("assistant\n\n")
        item['clarification_question'] = parts[-1].strip()
    if 'clarification_answer' in item:
        parts = item['clarification_answer'].split("assistant\n\n")
        item['clarification_answer'] = parts[-1].strip()
    if 'final_answer' in item:
        parts = item['final_answer'].split("assistant\n\n")
        item['final_answer'] = parts[-1].strip()
# write back to a new jsonl file
output_path = "./ca_base_dev_200_purified.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in data_list:
        f.write(json.dumps(item) + '\n') 