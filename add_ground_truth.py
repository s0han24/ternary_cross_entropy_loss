import json

def add_ground_truth_answers(results_file, ambigqa_file, output_file):
    """
    Add ground truth answers from AmbigQA data to results file.
    Creates duplicate rows if multiple answers exist (max 2 per question).
    
    Args:
        results_file: Path to results.jsonl
        ambigqa_file: Path to ambigqa.dev_4h.clarify.jsonl
        output_file: Path to output file with ground truth answers
    """
    # Load AmbigQA data and create a lookup dictionary
    ambigqa_data = {}
    with open(ambigqa_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            ambigqa_data[record['id']] = record.get('answers', [])
    
    # Process results file and add ground truth answers
    output_records = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            result_id = result['id']
            
            # Get ground truth answers for this ID
            if result_id in ambigqa_data:
                answers = ambigqa_data[result_id]
                # Take at most 2 answers
                answers_to_use = answers[:2] if len(answers) >= 2 else answers
                
                # Create a row for each answer
                if answers_to_use:
                    for answer in answers_to_use:
                        new_record = result.copy()
                        new_record['ground_truth_answer'] = answer
                        output_records.append(new_record)
                else:
                    # No answers available, add None
                    result['ground_truth_answer'] = None
                    output_records.append(result)
            else:
                # ID not found in AmbigQA data
                result['ground_truth_answer'] = None
                output_records.append(result)
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Processed {len(output_records)} records")
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    results_file = "cq_sft_dev_100.jsonl"
    ambigqa_file = "data/ambigqa.dev_4h.clarify.jsonl"
    output_file = "cq_sft_dev_200.jsonl"
    
    add_ground_truth_answers(results_file, ambigqa_file, output_file)
