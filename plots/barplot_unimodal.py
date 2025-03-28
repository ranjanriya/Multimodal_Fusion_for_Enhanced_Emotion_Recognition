import matplotlib.pyplot as plt
import numpy as np
import sys

# Define the metrics and model groups
metrics_labels = ['Train Accuracy', 'Test Accuracy']
group_labels = ['2D CNN', '3D CNN', 'Transformer']

# Specific encoders for each type
encoders = [
    ['ResNet18', 'GoogLeNet', 'VGG16'],
    ['Simple3D', 'Ablated Simple3D', 'I3D', 'Ablated I3D'],
    ['ViT', 'PT ViT', 'VideoMAE']
]

# Simulated data for each encoder, each with: [Train Accuracy, Test Accuracy]

# Audio
data_audio = {
    'ResNet18': [0.9848, 0.6182],
    'GoogLeNet': [0.9344, 0.6244],
    'VGG16': [0.9113, 0.5962],
    'Simple3D': [0.519, 0.514],
    'Ablated Simple3D': [0, 0.354],
    'I3D': [0.623, 0.605],
    'Ablated I3D' : [0, 0.448],
    'ViT': [0.4710, 0.4401],
    'PT ViT': [0.9846, 0.6231],
    'VideoMAE': [0.344 , 0.372]
}

# Vision
data_vision = {
    'ResNet18': [0.9848, 0.6815],
    'GoogLeNet': [0.9275, 0.6742],
    'VGG16': [0.9404, 0.6765],
    'Simple3D': [0.546, 0.462],
    'Ablated Simple3D': [0, 0.298],
    'I3D': [0.878, 0.831],
    'Ablated I3D' : [0, 0.540],
    'ViT': [0.7612, 0.5536],
    'PT ViT': [0.9857, 0.6895],
    'VideoMAE': [0.170, 0.188]
}

if sys.argv[1] == 'audio':
    data = data_audio
elif sys.argv[1] == 'vision':
    data = data_vision
else:
    print('Error')
    exit(1)

# colors = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#FF99CC', '#CCFF99', '#99FFFF']  # Colors for each encoder
colors = ['red', 'green', 'blue', 'cyan', 'teal', 'orange', 'yellow', 'purple', 'magenta', 'lime']  # Colors for each encoder
# hatches = ['/', '\\', 'o', 'x']  # Different hatches for each metric

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

metric_group_width = 0.8
bar_width = metric_group_width / (len(encoders[0]) + len(encoders[1]) + len(encoders[2]) + 1)
x = np.arange(len(metrics_labels)) * (len(encoders[0]) + len(encoders[1]) + len(encoders[2]) + 3) * bar_width
alpha = {0: 0.6, 1: 0.25, 2: 0.65}
# Generate the bars
for i, metric in enumerate(metrics_labels):
    offset = 0  # reset offset for each metric group
    col_count = 0
    for j, group in enumerate(group_labels):
        off = alpha[j]
        ax.text(x[i] + (j + off) * (len(encoders[j]) + 1) * bar_width - 0.5 * len(group_labels) * bar_width, -0.05,
                group, ha='center', va='center')
        for k, encoder in enumerate(encoders[j]):
            metric_index = i
            val = data[encoder][metric_index]
            # rect = ax.bar(x[i] + offset, val, bar_width, label=metric if j == 0 and k == 0 else "",
            #               color=colors[j * len(encoders[j]) + k], edgecolor='black', hatch=hatches[i])
            # rect = ax.bar(x[i] + offset, val, bar_width, label=metric if j == 0 and k == 0 else "",
            #               color=colors[j * len(encoders[j]) + k], edgecolor='black')
            rect = ax.bar(x[i] + offset, val, bar_width, label=metric if j == 0 and k == 0 else "",
                          color=colors[col_count], edgecolor='black')
            ax.text(rect[0].get_x() + rect[0].get_width() / 2, 0,
                    encoder, ha='center', va='bottom', rotation=90, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')) # , weight='bold'
            offset += bar_width
            col_count += 1
        offset += bar_width * 0.5  # Add spacer after each group
    offset += bar_width * 1.5  # Extra space between metrics

ax.set_ylim(0.0, 1.0)
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Metrics')
ax.set_ylabel('Values', fontsize=12)
# ax.set_title('Unimodal visual pipeline for different encoders')
ax.set_title('Unimodal ' + sys.argv[1] + ' for different encoders')
ax.set_xticks(x + (metric_group_width + 0.0 * bar_width) / 2)
ax.tick_params(axis='x', which='both', bottom=False)  # Remove x-tick markers
ax.set_xticklabels(metrics_labels, y=-0.05, fontsize=12)
# ax.legend([plt.Rectangle((0,0),1,1, color='gray', hatch=hatch) for hatch in hatches], metrics_labels, title='Metric')
# Add grid lines parallel to x-axis
ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')

# Add separator lines between each metric group
for i in range(len(metrics_labels) - 1):
    ax.axvline(x[i] + (len(encoders[2]) + 8.5) * bar_width, color='black', linestyle='-', linewidth=2)


# ax.axhline(1.0, color='red', linestyle='-', linewidth=2)
plt.savefig('unimodal-'+sys.argv[1]+'.png')
fig.tight_layout()
plt.show()
