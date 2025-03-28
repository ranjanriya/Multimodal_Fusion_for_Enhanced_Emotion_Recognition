import matplotlib.pyplot as plt
import numpy as np

# Define the metrics and model groups
metrics_labels = ['Train Accuracy', 'Test Accuracy']
group_labels = ['2D CNN', '3D CNN', 'Transformer']

# Specific encoders for each type
encoders = [
    ['ResNet18', 'GoogLeNet', 'VGG16'],
    # ['Simple3DCNN', 'I3D', 'Ablated I3D'],
    ['Simple3D', 'I3D'],
    ['ViT', 'PT ViT', 'VideoMAE']
]

# Simulated data for each encoder, each with: [Train Accuracy, Test Accuracy]

# Audio
# data = {
#     'ResNet18': [0.8216, 0.5825],
#     'GoogLeNet': [0, 0],# Placeholder values
#     'VGG16': [0.6366, 0.5120],
#     'Simple3DCNN': [0.519, 0.514],
#     'I3D': [0.623, 0.605],
#     'Ablated I3D' : [0, 0.448],
#     'ViT': [0.1634, 0.1738],
#     'VideoMAE': [0, 0]  # Placeholder values
# }

# Vision
# data = {
#     'ResNet18': [0.8634, 0.6225],
#     'GoogLeNet': [0, 0],# Placeholder values
#     'VGG16': [0.9495, 0.7040],
#     'Simple3DCNN': [0.546, 0.462],
#     'I3D': [0.878, 0.831],
#     'Ablated I3D' : [0, 0.540],
#     'ViT': [0.8361, 0.5934],
#     'VideoMAE': [0, 0]  # Placeholder values
# }

# Multimodal
data = {
    'ResNet18': [0.9379, 0.7326],
    'GoogLeNet': [0.9390, 0.6711],
    'VGG16': [0.9356, 0.6608],
    'Simple3D': [0.647, 0.573],
    'I3D': [0.923, 0.806],
    'ViT': [0.8002, 0.6909],
    'PT ViT': [0.9867, 0.8290],
    'VideoMAE': [0.334, 0.366]
}

# colors = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#FF99CC', '#CCFF99', '#99FFFF']  # Colors for each encoder
# colors = ['red', 'green', 'blue', 'cyan', 'orange', 'yellow', 'purple', 'black']  # Colors for each encoder
colors = ['red', 'green', 'blue', 'cyan', 'orange', 'purple', 'magenta', 'lime']  # Colors for each encoder
# hatches = ['/', '\\', 'o', 'x']  # Different hatches for each metric

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

metric_group_width = 0.8
bar_width = metric_group_width / (len(encoders[0]) + len(encoders[1]) + len(encoders[2]) + 1)
x = np.arange(len(metrics_labels)) * (len(encoders[0]) + len(encoders[1]) + len(encoders[2]) + 3) * bar_width
alpha = {0: 0.6, 1: 0.8, 2: 0.2}
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
ax.set_title('Multimodal for different encoders')
ax.set_xticks(x + (metric_group_width + 0.0 * bar_width) / 2)
ax.tick_params(axis='x', which='both', bottom=False)  # Remove x-tick markers
ax.set_xticklabels(metrics_labels, y=-0.05, fontsize=12)
# ax.legend([plt.Rectangle((0,0),1,1, color='gray', hatch=hatch) for hatch in hatches], metrics_labels, title='Metric')
# Add grid lines parallel to x-axis
ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')

# Add separator lines between each metric group
for i in range(len(metrics_labels) - 1):
    ax.axvline(x[i] + (len(encoders[2]) + 6.5) * bar_width, color='black', linestyle='-', linewidth=2)


# ax.axhline(1.0, color='red', linestyle='-', linewidth=2)

fig.tight_layout()
plt.savefig('multimodal.png')
plt.show()
