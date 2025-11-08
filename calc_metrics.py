import json
from evaluate import load
# Load metrics
rouge = load("rouge")
bleu = load("bleu")
bertscore = load("bertscore")
meteor = load("meteor")

# Load data
generations = []
with open('fa_base_dev_200_base.jsonl', 'r') as f:
    for line in f:
        generations.append(json.loads(line))

def apply_metrics(ref, pred):
    """Apply all metrics to a reference-prediction pair"""
    return {
        'rouge': rouge.compute(predictions=[pred], references=[ref]),
        'bleu': bleu.compute(predictions=[pred], references=[[ref]]),
        'bertscore': bertscore.compute(predictions=[pred], references=[ref], lang='en'),
        'meteor': meteor.compute(predictions=[pred], references=[ref])
    }

# Collect results
results = {
    'rouge': [],
    'bleu': [],
    'bertscore': [],
    'meteor': []
}

# Process each generation
for gen in generations:
    ref = gen['ground_truth_answer']
    pred = gen['final_answer']
    
    metrics = apply_metrics(ref, pred)
    
    results['rouge'].append(metrics['rouge'])
    results['bleu'].append(metrics['bleu'])
    results['bertscore'].append(metrics['bertscore'])
    results['meteor'].append(metrics['meteor'])

# Aggregate results
n = len(results['rouge'])
final_results = {
    'rouge': {
        'rouge1': sum(r['rouge1'] for r in results['rouge']) / n,
        'rouge2': sum(r['rouge2'] for r in results['rouge']) / n,
        'rougeL': sum(r['rougeL'] for r in results['rouge']) / n,
    },
    'bleu': {
        'bleu': sum(b['bleu'] for b in results['bleu']) / n,
    },
    'bertscore': {
        'precision': sum(b['precision'][0] for b in results['bertscore']) / n,
        'recall': sum(b['recall'][0] for b in results['bertscore']) / n,
        'f1': sum(b['f1'][0] for b in results['bertscore']) / n,
    },
    'meteor': {
        'meteor': sum(m['meteor'] for m in results['meteor']) / n,
    }
}

# Print results
print("Evaluation Results:")
print(json.dumps(final_results, indent=2))