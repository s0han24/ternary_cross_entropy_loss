import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results
with open('results.json', 'r') as f:
    results = json.load(f)

# Extract model names and metrics
models = list(results.keys())
metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'bertscore_f1', 'meteor']

# Prepare data for plotting
data = {metric: [] for metric in metrics}
for model in models:
    data['rouge1'].append(results[model]['rouge']['rouge1'])
    data['rouge2'].append(results[model]['rouge']['rouge2'])
    data['rougeL'].append(results[model]['rouge']['rougeL'])
    data['bleu'].append(results[model]['bleu']['bleu'])
    data['bertscore_f1'].append(results[model]['bertscore']['f1'])
    data['meteor'].append(results[model]['meteor']['meteor'])

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Evaluation Metrics Comparison', fontsize=16, fontweight='bold')

# Plot each metric
for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    values = data[metric]
    bars = ax.bar(range(len(models)), values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(models))))
    
    ax.set_title(metric.upper().replace('_', ' '), fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a heatmap for all metrics
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data matrix
metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore F1', 'METEOR']
data_matrix = np.array([data[m] for m in metrics])

# Create heatmap
im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(metric_names)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_yticklabels(metric_names)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score', rotation=270, labelpad=20)

# Add text annotations
for i in range(len(metric_names)):
    for j in range(len(models)):
        text = ax.text(j, i, f'{data_matrix[i, j]:.4f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('metrics_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Create grouped bar chart for ROUGE metrics
fig, ax = plt.subplots(figsize=(12, 6))

rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
x = np.arange(len(models))
width = 0.25

for i, metric in enumerate(rouge_metrics):
    values = data[metric]
    ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('ROUGE Metrics Comparison Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('rouge_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved as:")
print("- metrics_comparison.png")
print("- metrics_heatmap.png")
print("- rouge_comparison.png")