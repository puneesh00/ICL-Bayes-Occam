import torch
import os
os.environ['MPLCONFIGDIR'] = "/scratch/st-cthrampo-1/puneesh"
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import CustomTransformer
#from utils import generate_test_sequences
from train import train_transformer_model, train_transformer_model_new
from gpt import GPT, gpt, create_padding_mask
from train_gpt import train_gpt_model, train_transformer_gpt, train_transformer_gpt_hf
from utils import open_log_files, visualize_and_save_attention, load_model_from_directory, tensor_to_nested_dict_with_keys
import argparse

from utils import MarkovChainGenerator, MarkovChainEvaluator, SequenceAnalyzer, combine_stats
from linear_probe import extract_layer_representation, train_regression_head_for_layer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for transformer model.")

    # Add arguments
    parser.add_argument('--sequence_length', type=int, default=100, help="Length of sequence (default: 100)")
    #parser.add_argument('--bs', type=int, default=16, help="Bs for training (default: 16)")
    parser.add_argument('--dim', type=int, default=16, help="embed dimension (default: 16)")

    parser.add_argument('--path', type=str, required=True, help="model directory and path to save file")
    parser.add_argument('--num_heads', type=int, default=2, help="number of heads")
    parser.add_argument('--num_layers', type=int, default=2, help="number of encoder layers")
    #parser.add_argument('--autoregressive', action='store_true', help="Enable autoregressive training")
    parser.add_argument('--if_layer_norm', action='store_true', help='Use layer norm in the transformer encoder')
    parser.add_argument('--if_mlp', action='store_true', help='Use mlp in the transformer encoder')

    #parser.add_argument('--if_dropout', action='store_true', help='Use dropout in the transformer encoder')
    parser.add_argument('--rpe', action='store_true', help='Use relative PE in transformer encoder')
    parser.add_argument('--vocab_size', type=int, default=3, help="Vocab size")
    parser.add_argument('--sparsity', type=float, default=1, help="Sparsity in transitions")
    parser.add_argument('--fix_seq_len', action='store_true', help='Seq length fixed or not')

    parser.add_argument(
        '--test_order_list',
        type=str,            
        default="1,2,3",      
        help="order of the chains during training"
    )

    parser.add_argument(
        '--test_seq_len',
        type=str,            
        default="200,500",      
        help="len of sequences during test"
    )
    # Parse arguments
    args = parser.parse_args()
    return args


# Parse the arguments
args = parse_arguments()

# Access argument values

sequence_length = args.sequence_length

num_heads = args.num_heads
num_layers = args.num_layers
if_layer_norm = args.if_layer_norm
if_mlp = args.if_mlp

rpe = args.rpe
vocab_size = args.vocab_size
sparsity = args.sparsity
dim = args.dim
#fix_seq_len = args.fix_seq_len

test_seq_len = args.test_seq_len

#order_list = args.order_list.split(',')
#order_list = [int(x) for x in order_list]


test_order_list = args.test_order_list.split(',')
test_order_list = [int(x) for x in test_order_list]
max_order = max(test_order_list)
t_gram_list = list(range(0, max_order+2))

test_seq_len = args.test_seq_len.split(',')
test_seq_len = [int(x) for x in test_seq_len]


max_transitions = int(vocab_size*sparsity)

# Parameters
embed_dim = dim

lambda_ = 0.80402
combine_idx = [2, 4]

tgrams = ['Uniform', 'Unigram', 'Bigram', 'Trigram', 
                'Tetragram', 'Quadragram', 'Pentragram', 'Hexagram',
                'Heptagram', 'Octagram'] 

tgram_string = tgrams[0:t_gram_list[-1]+1]


model_seed = 0
torch.manual_seed(model_seed)


#Define the configuration KARPATHY's
config = GPT.get_default_config()
config.model_type = None  # We're not using a pre-defined model type
config.n_layer = num_layers
config.n_head = num_heads
config.n_embd = dim
config.vocab_size = vocab_size  # Replace with your actual vocabulary size
config.block_size = max_order*sequence_length # Replace with your desired sequence length
config.if_layer_norm = if_layer_norm
config.if_mlp = if_mlp

#Create the model instance this is Karpathy
save_path = args.path

transformer = GPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print("device", device)
transformer = transformer.to(device)

transformer = load_model_from_directory(save_path, transformer, device)


#(Optional) Set the model to evaluation mode
transformer.eval()

num_sequences = 1000
evaluator = MarkovChainEvaluator(vocab_size=vocab_size, max_transitions=max_transitions, structure=False, 
                                 orders=test_order_list, 
                                 num_sequences=num_sequences, sequence_length=test_seq_len, 
                                 t_grams=t_gram_list, seed=143432)

test_sequences = evaluator.generate_sequences_for_orders(base_seed=143432,stationary=True)  

split = {0, 1, 2, 3, 4, 5, 6, 7}
split_probs = {}
split_stats = {}

split_negll = {}
split_nzr = {}

split_len = test_seq_len[0]//len(split)

for s in split:
    #print("test sequences", test_sequences)
    t_gram_stats = {}
    t_gram_probs = {}
    t_gram_pi = {}
    non_zero_rows = {}
    for ord in test_order_list:
        print("split", s)
        print("ord", ord)
        print("test_seq_shape", test_sequences[ord][:,0:(s+1)*split_len].shape)
        t_gram_stats[ord], t_gram_probs[ord], t_gram_pi[ord] = evaluator.compute_t_gram_stats(test_sequences[ord][:,0:(s+1)*split_len]) 
        
        # Get a tensor to track zero rows for all t-grams
        zero_rows_union = torch.zeros(t_gram_stats[ord][1].shape[0], dtype=torch.bool)  # Initialize with all False

        # Check for zero rows in all t-gram statistics for the given order
        for t in t_gram_list[1:]:
            zero_rows_union |= (t_gram_stats[ord][t] == 0).all(dim=1)  # Union of zero rows

        # Non-zero rows
        non_zero_rows[ord] = ~zero_rows_union

        # Filter out rows with all zeros in each t-gram statistic
        for t in t_gram_list[1:]:
            t_gram_stats[ord][t] = t_gram_stats[ord][t][non_zero_rows[ord]]
            t_gram_probs[ord][t] = t_gram_probs[ord][t][non_zero_rows[ord]]

    split_probs[s] = t_gram_probs
    split_stats[s] = t_gram_stats
    Negll = np.zeros((len(test_order_list), test_order_list[-1], num_sequences))


    split_nzr[s] = non_zero_rows
    
    negll_ord = {}
    for idx, ord in enumerate(test_order_list):
        negll = {}
        for t in t_gram_list[1:-1]:
            #print("shape of seqs", test_sequences[ord][split_nzr[s][ord],0:(s+1)*100].shape)
            #print("shape of probs", split_probs[s][ord][t+1].shape)
            M = SequenceAnalyzer.compute_avg_neg_log_likelihood(test_sequences[ord][split_nzr[s][ord],0:(s+1)*split_len], split_probs[s][ord][t+1], t, vocab_size)
            negll[t+1] = M
        negll_ord[ord] = negll
    split_negll[s] = negll_ord    
results_order = {}

results_select = {}

select_stats = combine_stats(split_stats, split_negll, split_nzr, test_order_list, vocab_size, lambda_, combine_idx)
#print("select_stats", select_stats)

#print("split_stats", split_stats)
with torch.no_grad():
    for ord in test_order_list:

        results = {}

        results_sel = {}
        test_data = test_sequences[ord].to(device)
        #print(test_data.shape)
        test_outputs,_ = transformer(test_data)
        #print(test_outputs.shape)
        for s in split:
                #print("keys", split_stats[s][ord].keys())
                #print("values", split_stats[s][ord].values())
                idx = (s+1)*split_len-1
                test_o = test_outputs[:,idx,:]
                #print(f"non-zero-rows for split s:{s} is {split_nzr[s][ord].sum().item()}")
                test_o = test_o[split_nzr[s][ord]]
                #if ord == 1:
                    #test_outputs = test_outputs[non_zero_rows]
                results[s] = evaluator.run_evaluation(test_o, split_stats[s][ord])   #Dict[int (ord), Dict[int (t-gram), float]]

                #results_sel[s] = evaluator.run_evaluation(select_stats[s][ord], split_stats[s][ord], pred=True)

        results_order[ord] = results    
        #results_select[ord] = results_sel
#for key, value in results_order.items():
   #print(f"results for order {key}: {value}")

#print("results selection rule", results_select)

p = len(split)  # Number of splits
split_lengths = [test_seq_len[0] // p * (i + 1) for i in range(p)]  # [100, 200, 300, 400]

# Map distribution indices to descriptive labels (optional)
distribution_labels = {
    0: "Uniform",
    1: "Unigram",
    2: "Bigram",
    3: "Trigram",
    4: "Tetragram",
    5: "Pentagram"
}

#Define a color palette for the distributions
num_distributions = len(results_order[1][0].keys())  # Number of distributions
color_palette = plt.cm.tab10(np.linspace(0, 1, num_distributions))  # Adjust as needed

# Map each dist_idx to a color
dist_to_color = {dist_idx: color_palette[i] for i, dist_idx in enumerate(results_order[1][0].keys())}


# # Updated figure layout
# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
# #fig.suptitle("KL Distances to Different Distributions by Sequence Length", fontsize=16)

# # Flatten axes for easy iteration
# axes = axes.flatten()

# # Ensure we have enough subplots for the orders
# if len(test_order_list) > len(axes):
#     raise ValueError("Too many orders for the current subplot layout. Adjust the rows/columns.")



# # Plot each order in a subplot
# for idx, order in enumerate(test_order_list):
#     ax = axes[idx]
#     for dist_idx in results_order[order][0].keys():  # Loop through distribution indices
#         # Extract distances for the current distribution across splits
#         distances = [results_order[order][split][dist_idx] for split in range(p)]

#         distance_sel = [results_select[order][split][dist_idx] for split in range(p)]
#         label = distribution_labels.get(dist_idx, f"Dist {dist_idx}")  # Use labels or default

#         color = dist_to_color[dist_idx]
#         ax.plot(split_lengths, distances, marker="o", color=color, label=label)
#         ax.plot(split_lengths, distance_sel, marker='x', color=color, label=label)

#     # Formatting the subplot
#     ax.set_title(f"Order {order}", fontsize=12)
#     ax.set_xlabel("Sequence Length", fontsize=10)
#     ax.set_ylabel("KL Distance", fontsize=10)
#     ax.legend(title="Distribution", fontsize=8)
#     ax.grid(True)

# # Remove unused subplots (if test_order_list has fewer than 4 orders)
# for extra_ax in axes[len(test_order_list):]:
#     fig.delaxes(extra_ax)

# # Adjust layout
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Accommodate the suptitle

# # Save the plot
# save_it = os.path.join(save_path, "kl_distance_plot_by_len_combined_sum_wtrigram.png")
# plt.savefig(save_it, dpi=300, bbox_inches="tight")  # Save as PNG

# plt.show()

# Updated figure layout
# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
# #fig.suptitle("KL Distances to Different Distributions by Sequence Length", fontsize=16)

# # Flatten axes for easy iteration
# axes = axes.flatten()
# # Plot each order in a subplot
# for idx, order in enumerate(test_order_list):
#     ax = axes[idx]
#     for dist_idx in results_order[order][0].keys():  # Loop through distribution indices
#         # Extract distances for the current distribution across splits
#         distances = [results_order[order][split][dist_idx] for split in range(p)]

#         distance_sel = [results_select[order][split][dist_idx] for split in range(p)]
#         label = distribution_labels.get(dist_idx, f"Dist {dist_idx}")  # Use labels or default

#         color = dist_to_color[dist_idx]
#         #ax.plot(split_lengths, distances, marker="o", color=color, label=label)
#         ax.plot(split_lengths, distance_sel, marker='x', color=color, label=label)

#     # Formatting the subplot
#     ax.set_title(f"Order {order}", fontsize=12)
#     ax.set_xlabel("Sequence Length", fontsize=10)
#     ax.set_ylabel("KL Distance", fontsize=10)
#     ax.legend(title="Distribution", fontsize=8)
#     ax.grid(True)

# # Remove unused subplots (if test_order_list has fewer than 4 orders)
# for extra_ax in axes[len(test_order_list):]:
#     fig.delaxes(extra_ax)

# # Adjust layout
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Accommodate the suptitle

# # Save the plot
# save_it = os.path.join(save_path, "kl_distance_plot_by_len_select_rule_sum_wtrigram.png")
# plt.savefig(save_it, dpi=300, bbox_inches="tight")  # Save as PNG

# plt.show()


# Updated figure layout
fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)
#fig.suptitle("KL Distances to Different Distributions by Sequence Length", fontsize=16)

# Flatten axes for easy iteration
axes = axes.flatten()
# Plot each order in a subplot
for idx, order in enumerate(test_order_list):
    ax = axes[idx]
    for dist_idx in results_order[order][0].keys():  # Loop through distribution indices
        # Extract distances for the current distribution across splits
        distances = [results_order[order][split][dist_idx] for split in range(p)]

        #distance_sel = [results_select[order][split][dist_idx] for split in range(p)]
        label = distribution_labels.get(dist_idx, f"Dist {dist_idx}")  # Use labels or default

        color = dist_to_color[dist_idx]
        ax.plot(split_lengths, distances, marker="o", color=color, label=label)
        #ax.plot(split_lengths, distance_sel, marker='x', color=color, label=label)

    # Formatting the subplot
    ax.set_title(f"Order {order}", fontsize=12)
    ax.set_xlabel("Sequence Length", fontsize=10)
    ax.set_ylabel("KL Distance", fontsize=10)
    ax.legend(title="Distribution", fontsize=8)
    ax.grid(True)

# Remove unused subplots (if test_order_list has fewer than 4 orders)
for extra_ax in axes[len(test_order_list):]:
    fig.delaxes(extra_ax)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Accommodate the suptitle

# Save the plot
save_it = os.path.join(save_path, "kl_distance_plot_by_len_model_sum.png")
plt.savefig(save_it, dpi=300, bbox_inches="tight")  # Save as PNG

plt.show()


# Updated figure layout
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
#fig.suptitle("KL Distances to Different Distributions by Sequence Length", fontsize=16)

axes = axes.flatten()

# Process and plot for each order
for idx, order in enumerate(test_order_list):
    ax = axes[idx]
    
    # Prepare averages and std deviations for each statistic
    for dist_idx in split_negll[0][1].keys(): 
        averages = []
        std_devs = []
        
        # Compute averages and std deviations across splits
        for s in split:
            values = split_negll[s][order][dist_idx]  # Extract values
            values = values[values > 0].numpy()  # Ignore zeros
            avg = np.mean(values) if len(values) > 0 else 0
            std = np.std(values) if len(values) > 0 else 0
            averages.append(avg)
            std_devs.append(std)
        
        # Plot averages with shaded standard deviation region
        averages = np.array(averages)
        std_devs = np.array(std_devs)
        label = distribution_labels.get(dist_idx, f"Stat {dist_idx}")  # Use labels or default
        ax.plot(split_lengths, averages, label=label, marker="o")
        ax.fill_between(split_lengths, averages - std_devs, averages + std_devs, alpha=0.2)

    # Formatting the subplot
    ax.set_title(f"Order {order + 1}", fontsize=12)
    ax.set_xlabel("Sequence Length", fontsize=10)
    ax.set_ylabel("Neg Log-Likelihood", fontsize=12)
    ax.legend(title="Statistic", fontsize=10)
    ax.grid(True)


save_it = os.path.join(save_path, "neg_ll_by_len_sum.png")

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to accommodate the suptitle
plt.savefig(save_it, dpi=300, bbox_inches="tight")  # Save as PNG
plt.show()