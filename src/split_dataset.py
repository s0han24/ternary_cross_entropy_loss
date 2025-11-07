import json
import random
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def split_dataset(input_file, output_dir, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=42):
    # Ensure ratios sum to 1
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_jsonl(input_file)
    
    # Shuffle data
    random.seed(seed)
    random.shuffle(data)
    
    # Calculate split indices
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    dev_size = int(total_size * dev_ratio)
    
    # Split data
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]
    
    # Save splits
    save_jsonl(train_data, Path(output_dir) / 'final_train.jsonl')
    save_jsonl(dev_data, Path(output_dir) / 'final_dev.jsonl')
    save_jsonl(test_data, Path(output_dir) / 'final_test.jsonl')
    
    # Print statistics
    print(f"Total examples: {total_size}")
    print(f"Train split: {len(train_data)} examples")
    print(f"Dev split: {len(dev_data)} examples")
    print(f"Test split: {len(test_data)} examples")

if __name__ == "__main__":
    input_file = "/scratch/sujayb/anlp/clarifying_questions/data/final_clarify.jsonl"
    output_dir = "/scratch/sujayb/anlp/clarifying_questions/data/splits"
    
    split_dataset(input_file, output_dir)