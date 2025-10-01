import json
import random
import uuid
from typing import List, Dict, Any, Tuple

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_type1_entries(ambigqa_data: List[Dict[str, Any]], count: int = 1000) -> List[Dict[str, Any]]:
    """
    Type 1: AmbigQA entries with multipleQAs - choose random answer (Presumptuous answer)
    Label: 0 (Bad response - presumptuous answer to ambiguous question)
    """
    type1_entries = []
    
    # Filter entries with multipleQAs
    multiple_qa_entries = [
        entry for entry in ambigqa_data 
        if entry.get('annotations') and 
        any(ann.get('type') == 'multipleQAs' for ann in entry['annotations'])
    ]
    
    print(f"Found {len(multiple_qa_entries)} entries with multipleQAs")
    
    for i, entry in enumerate(multiple_qa_entries[:count]):
        # Find the multipleQAs annotation
        multiple_qa_ann = None
        for ann in entry['annotations']:
            if ann.get('type') == 'multipleQAs':
                multiple_qa_ann = ann
                break
        
        if multiple_qa_ann and 'qaPairs' in multiple_qa_ann:
            # Choose a random QA pair
            qa_pair = random.choice(multiple_qa_ann['qaPairs'])
            answer = qa_pair['answer'][0] if qa_pair['answer'] else "No answer available"
            
            dataset_entry = {
                'id': f"type1_{i+1:04d}",
                'prompt': entry['question'],
                'candidate_response': answer,
                'label': 0,  # Presumptuous answer to ambiguous question
                'type': 'type1',
                'source': 'AmbigQA_multipleQAs'
            }
            type1_entries.append(dataset_entry)
    
    return type1_entries

def generate_type2_entries(cambignq_data: List[Dict[str, Any]], count: int = 1000) -> List[Dict[str, Any]]:
    """
    Type 2: CAmbigNQ entries - use clarifying questions
    Label: 1 (Good response - clarifying question for ambiguous question)
    """
    type2_entries = []
    
    # Filter entries that have clarification questions
    clarifying_entries = [
        entry for entry in cambignq_data 
        if entry.get('clarification_question')
    ]
    
    print(f"Found {len(clarifying_entries)} entries with clarification questions")
    
    for i, entry in enumerate(clarifying_entries[:count]):
        dataset_entry = {
            'id': f"type2_{i+1:04d}",
            'prompt': entry['question'],
            'candidate_response': entry['clarification_question'],
            'label': 1,  # Good response - clarifying question
            'type': 'type2',
            'source': 'CAmbigNQ_clarifying'
        }
        type2_entries.append(dataset_entry)
    
    return type2_entries

def generate_type3_entries(ambigqa_data: List[Dict[str, Any]], count: int = 1000) -> List[Dict[str, Any]]:
    """
    Type 3: AmbigQA entries with singleAnswer - direct answers
    Label: 1 (Good response - direct answer to unambiguous question)
    """
    type3_entries = []
    
    # Filter entries with singleAnswer
    single_answer_entries = [
        entry for entry in ambigqa_data 
        if entry.get('annotations') and 
        any(ann.get('type') == 'singleAnswer' for ann in entry['annotations'])
    ]
    
    print(f"Found {len(single_answer_entries)} entries with singleAnswer")
    
    for i, entry in enumerate(single_answer_entries[:count]):
        # Find the singleAnswer annotation
        single_answer_ann = None
        for ann in entry['annotations']:
            if ann.get('type') == 'singleAnswer':
                single_answer_ann = ann
                break
        
        if single_answer_ann and 'answer' in single_answer_ann:
            answer = single_answer_ann['answer'][0] if single_answer_ann['answer'] else "No answer available"
            
            dataset_entry = {
                'id': f"type3_{i+1:04d}",
                'prompt': entry['question'],
                'candidate_response': answer,
                'label': 1,  # Good response - direct answer to unambiguous question
                'type': 'type3',
                'source': 'AmbigQA_singleAnswer'
            }
            type3_entries.append(dataset_entry)
    
    return type3_entries

def create_dataset(ambigqa_file: str, cambignq_file: str, 
                  type1_count: int = 1000, type2_count: int = 1000, type3_count: int = 1000) -> List[Dict[str, Any]]:
    """
    Create the complete dataset by combining all three types
    """
    print("Loading data files...")
    
    # Load the datasets
    ambigqa_data = load_json(ambigqa_file)
    cambignq_data = load_json(cambignq_file)
    
    print(f"Loaded {len(ambigqa_data)} AmbigQA entries")
    print(f"Loaded {len(cambignq_data)} CAmbigNQ entries")
    
    # Generate each type of entry
    print("\nGenerating Type 1 entries (Presumptuous answers)...")
    type1_entries = generate_type1_entries(ambigqa_data, type1_count)
    
    print("\nGenerating Type 2 entries (Clarifying questions)...")
    type2_entries = generate_type2_entries(cambignq_data, type2_count)
    
    print("\nGenerating Type 3 entries (Direct answers)...")
    type3_entries = generate_type3_entries(ambigqa_data, type3_count)
    
    # Combine all entries
    dataset = type1_entries + type2_entries + type3_entries
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    print(f"\nDataset created successfully!")
    print(f"Type 1 (label=0): {len(type1_entries)} entries")
    print(f"Type 2 (label=1): {len(type2_entries)} entries") 
    print(f"Type 3 (label=1): {len(type3_entries)} entries")
    print(f"Total: {len(dataset)} entries")
    
    # Calculate label distribution
    label_0_count = sum(1 for entry in dataset if entry['label'] == 0)
    label_1_count = sum(1 for entry in dataset if entry['label'] == 1)
    print(f"Label distribution: 0={label_0_count}, 1={label_1_count}")
    
    return dataset

def main():
    """Main function to create and save the dataset"""
    # Set random seed for reproducibility
    random.seed(42)
    
    # File paths
    ambigqa_file = "train_light.json"
    cambignq_file = "cq_dev.json"
    output_file = "dataset.json"
    
    # Create dataset
    dataset = create_dataset(
        ambigqa_file=ambigqa_file,
        cambignq_file=cambignq_file,
        type1_count=1000,
        type2_count=1000, 
        type3_count=1000
    )
    
    # Save the dataset
    save_json(dataset, output_file)
    print(f"\nDataset saved to {output_file}")
    
    # Print some sample entries
    print("\nSample entries:")
    for i, entry in enumerate(dataset[:3]):
        print(f"\nEntry {i+1}:")
        print(f"  ID: {entry['id']}")
        print(f"  Type: {entry['type']}")
        print(f"  Prompt: {entry['prompt'][:100]}...")
        print(f"  Response: {entry['candidate_response'][:100]}...")
        print(f"  Label: {entry['label']}")

if __name__ == "__main__":
    main()