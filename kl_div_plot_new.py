import os
os.environ['MPLCONFIGDIR'] = "/scratch/st-cthrampo-1/puneesh"


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import argparse

parser = argparse.ArgumentParser(description="Training script for transformer model.")
parser.add_argument('--path', type=str, required=True, help="Main directory")
parser.add_argument('--order', type=str, default=3, help="max order")

args = parser.parse_args()

path = args.path
order = int(args.order)


# Update font settings for better readability and thickness
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 16,  # Increased font size
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "axes.linewidth": 1.5,  # Thicker axis lines
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,  # Thicker lines
})

# Load the data from the log file
def load_data(filename, order):
    steps = []
    loss = []
    kl_uniform = []
    kl_unigram = []
    kl_bigram = []
    kl_trigram = []
    kl_tetragram = []
    
    with open(filename, "r") as f:
        for line in f:
            if "Step" in line:
                # Extract the relevant data
                parts = line.split(',')
                step = int(parts[0].split()[1])
                l = float(parts[1].split()[1])
                kl_u = float(parts[3].split()[2])
                kl_ug = float(parts[4].split()[2])
                kl_bg = float(parts[5].split()[2])
                kl_tg = float(parts[6].split()[2])
                if order>2:
                    kl_ttg = float(parts[7].split()[2])
                else:
                    kl_ttg = 0

                steps.append(step)
                loss.append(l)
                kl_uniform.append(kl_u)
                kl_unigram.append(kl_ug)
                kl_bigram.append(kl_bg)
                kl_trigram.append(kl_tg)
                kl_tetragram.append(kl_ttg)
    
    return np.array(steps), np.array(loss), np.array(kl_uniform), np.array(kl_unigram), np.array(kl_bigram), np.array(kl_trigram), np.array(kl_tetragram)

# Plot two KL divergence curves side by side
def plot_two_kl_divergence(file1, file2, save_path, order, batch_size=32):
    # Load data from both files
    steps1, _, kl_uniform1, kl_unigram1, kl_bigram1, kl_trigram1, kl_tetragram1 = load_data(file1, order)
    steps2, _, kl_uniform2, kl_unigram2, kl_bigram2, kl_trigram2, kl_tetragram2 = load_data(file2, order)
    
    # Calculate "Sequences Seen" for both datasets
    sequences_seen1 = (steps1 * batch_size) / 1000  # Divide by 1,000 for clarity
    sequences_seen2 = (steps2 * batch_size) / 1000

    # Subsample data for smoother curves
    subsample_factor = 10  # Increase subsampling for smoother curves
    sequences_seen1 = sequences_seen1[::subsample_factor]
    kl_uniform1 = kl_uniform1[::subsample_factor]
    kl_unigram1 = kl_unigram1[::subsample_factor]
    kl_bigram1 = kl_bigram1[::subsample_factor]
    kl_trigram1 = kl_trigram1[::subsample_factor]
    kl_tetragram1 = kl_tetragram1[::subsample_factor]
    
    sequences_seen2 = sequences_seen2[::subsample_factor]
    kl_uniform2 = kl_uniform2[::subsample_factor]
    kl_unigram2 = kl_unigram2[::subsample_factor]
    kl_bigram2 = kl_bigram2[::subsample_factor]
    kl_trigram2 = kl_trigram2[::subsample_factor]
    kl_tetragram2 = kl_tetragram2[::subsample_factor]
    
    # Define custom colors for the curves
    #custom_colors = ['#c7e9b4', '#41b6c4', '#225ea8', '#081d58']
    #custom_colors = ['#fcc5c0', '#f768a1', '#dd3497', '#7a0177'] 
    #fcc5c0 #fa9fb5 #f768a1 #dd3497 #ae017e #7a0177 #49006a
    #custom_colors = ['#fcc5c0', '#f768a1', '#ae017e', '#49006a']
    #custom_colors = ['#ffffcc', '#fed976', '#fd8d3c', '#e31a1c', '#800026']
    #custom_colors = ['#edf8b1', '#7fcdbb', '#41b6c4', '#1d91c0', '#253494']

    #custom_colors = ['#fcc5c0', '#f768a1', '#ae017e', '#7a0177', '#49006a']
    #c7e9b4 #7fcdbb #41b6c4 #1d91c0 #225ea8 #253494 #081d58
    custom_colors = ['#edf8b1', '#c7e9b4', '#41b6c4', '#225ea8', '#081d58']
    #custom_colors = ['#c7e9b4', '#7fcdbb', '#1d91c0', '#225ea8', '#081d58']

    labels = ['Uniform', 'Unigram', 'Bigram', 'Trigram', 'Tetragram']

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)  # Reduced figure size

    # Plot data for the first file
    axes[0].plot(sequences_seen1, kl_uniform1, label=labels[0], color=custom_colors[0], linewidth=2.5)
    axes[0].plot(sequences_seen1, kl_unigram1, label=labels[1], color=custom_colors[1], linewidth=2.5)
    axes[0].plot(sequences_seen1, kl_bigram1, label=labels[2], color=custom_colors[2], linewidth=2.5)
    axes[0].plot(sequences_seen1, kl_trigram1, label=labels[3], color=custom_colors[3], linewidth=2.5)
    if order>2:
        axes[0].plot(sequences_seen1, kl_tetragram1, label=labels[4], color=custom_colors[4], linewidth=2.5)
    axes[0].set_xlabel("Sequences Seen (in thousands)", fontsize=16)
    axes[0].set_ylabel("KL-Div(Distribution || Model)", fontsize=16)
    axes[0].set_title("Order 1", fontsize=14)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].tick_params(axis='both', labelsize=14)
    
    # Plot data for the second file
    axes[1].plot(sequences_seen2, kl_uniform2, label=labels[0], color=custom_colors[0], linewidth=2.5)
    axes[1].plot(sequences_seen2, kl_unigram2, label=labels[1], color=custom_colors[1], linewidth=2.5)
    axes[1].plot(sequences_seen2, kl_bigram2, label=labels[2], color=custom_colors[2], linewidth=2.5)
    axes[1].plot(sequences_seen2, kl_trigram2, label=labels[3], color=custom_colors[3], linewidth=2.5)
    if order>2:
        axes[1].plot(sequences_seen2, kl_tetragram2, label=labels[4], color=custom_colors[4], linewidth=2.5)
        axes[1].set_title("Order 3", fontsize=14)
    else:
        axes[1].set_title("Order 2", fontsize=14)
    axes[1].set_xlabel("Sequences Seen (in thousands)", fontsize=16)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axes[1].tick_params(axis='both', labelsize=14)

    fig.legend(labels,loc="upper center",ncol=5,  # Make the legend span all 5 labels in one row
    fontsize=14,frameon=False,bbox_to_anchor=(0.5, 1.02)  # Bring the legend closer to the plots
    )

    # Adjust layout to make space for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Reduced top space for the legend

    # Save and show the plot
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
path1 = os.path.join(path, 'test_kl_divergence_log_order1.txt')
if order>2:
    path2 = os.path.join(path, 'test_kl_divergence_log_order3.txt')
else:
    path2 = os.path.join(path, 'test_kl_divergence_log_order2.txt')


plot_two_kl_divergence(
    file1=path1,
    file2=path2,
    save_path=os.path.join(path, 'nice_plot.png'),
    order=order,
    batch_size=32
)