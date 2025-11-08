import json
from google import genai
from time import sleep

API_KEY = "AIzaSyDIm12mmFZYYlQ-K0zQOAD6DLrGVu-FDxU"  # Replace with your actual API key
client = genai.Client(api_key=API_KEY)

def generate_clarification_answer(question, clarification_question, ground_truth_answer):
    """
    Generate a clarification answer from Gemini acting as a user.
    """
    prompt = f"""You are a user who asked the following question:
"{question}"

You received this clarification question:
"{clarification_question}"

You are looking for this specific answer: "{ground_truth_answer}"

IMPORTANT: Do NOT reveal or mention the ground truth answer "{ground_truth_answer}" in your response. Instead, provide a brief, natural response to the clarification question that guides toward the answer you're seeking WITHOUT directly stating it. Respond as if you were the original user seeking information."""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return response.text

# Read results from results.jsonl
def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list

# Main processing
input_file = "cq_base_dev_200.jsonl"
output_file = "cq_ca_base_dev_200.jsonl"

print(f"Reading from {input_file}...")
data_list = read_jsonl(input_file)
print(f"Loaded {len(data_list)} examples")

results_with_answers = []

try:
    for idx, item in enumerate(data_list[:200]):
        print(f"Processing example {idx + 1}/{len(data_list)}...")

        question = item['question']
        clarification_question = item['clarification_question']
        ground_truth_answer = item['ground_truth_answer']

        # Generate clarification answer using Gemini
        clarification_answer = generate_clarification_answer(question, clarification_question, ground_truth_answer)

        # Create new entry with all fields including clarification_answer
        new_item = {
            "id": item['id'],
            "question": item['question'],
            "clarification_question": item['clarification_question'],
            "ground_truth_answer": item['ground_truth_answer'],
            "clarification_answer": clarification_answer
        }

        results_with_answers.append(new_item)

        # Print sample
        if idx < 3:
            print(f"\n--- Sample {idx + 1} ---")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth_answer}")
            print(f"Clarification Question: {clarification_question}")
            print(f"Clarification Answer: {clarification_answer}")
        sleep(6)
except Exception as e:
    print("bruh")

# Save to results_2.jsonl
print(f"\nSaving results to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for item in results_with_answers:
        f.write(json.dumps(item) + '\n')

print(f"\nDone! Saved {len(results_with_answers)} examples to {output_file}")