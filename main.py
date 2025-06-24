import torch
import os
os.environ['MPLCONFIGDIR'] = "/scratch/st-cthrampo-1/puneesh"
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gpt import GPT, gpt
from train_gpt import train_transformer_gpt
from utils import open_log_files
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for transformer model.")

    # Add arguments
    parser.add_argument('--lr', type=float, default=0.001, help="LR for the optimizer (default: 0.001)")
    parser.add_argument('--steps', type=int, default=100, help="Num steps for training (default: 100)")
    parser.add_argument('--sequence_length', type=int, default=100, help="Length of sequence (default: 100)")
    parser.add_argument('--bs', type=int, default=16, help="Bs for training (default: 16)")
    parser.add_argument('--dim', type=int, default=16, help="embed dimension (default: 16)")

    parser.add_argument('--log_interval', type=int, default=10, help="Log after every blah step(default: 10)")
    parser.add_argument('--save_interval', type=int, default=10, help="save after every blah step(default: 10)")
    parser.add_argument('--path', type=str, required=True, help="Path to save file")
    parser.add_argument('--num_heads', type=int, default=2, help="number of heads")
    parser.add_argument('--num_layers', type=int, default=2, help="number of encoder layers")

    parser.add_argument('--if_layer_norm', action='store_true', help='Use layer norm in the transformer encoder')
    parser.add_argument('--if_mlp', action='store_true', help='Use mlp in the transformer encoder')
    
    parser.add_argument('--vocab_size', type=int, default=3, help="Vocab size")
    parser.add_argument('--sparsity', type=float, default=1, help="Sparsity in transitions")

    parser.add_argument(
        '--order_list',
        type=str,            
        default="1,3",      
        help="order of the chains during training"
    )

    parser.add_argument(
        '--test_order_list',
        type=str,            
        default="1,3",      
        help="order of the chains during test"
    )

    parser.add_argument(
        '--test_seq_len',
        type=str,            
        default="300,300",      
        help="len of sequences during test"
    )
    # Parse arguments
    args = parser.parse_args()
    return args


# Parse the arguments
args = parse_arguments()

# Access argument values
lr = args.lr
steps = args.steps
sequence_length = args.sequence_length
bs = args.bs
log_interval = args.log_interval
save_interval = args.save_interval
save_path = args.path

num_heads = args.num_heads
num_layers = args.num_layers
if_layer_norm = args.if_layer_norm
if_mlp = args.if_mlp
vocab_size = args.vocab_size
sparsity = args.sparsity

dim = args.dim

test_seq_len = args.test_seq_len

order_list = args.order_list.split(',')
order_list = [int(x) for x in order_list]
max_order = max(order_list)
t_gram_list = list(range(0, max_order+3))

test_order_list = args.test_order_list.split(',')
test_order_list = [int(x) for x in test_order_list]

test_seq_len = args.test_seq_len.split(',')
test_seq_len = [int(x) for x in test_seq_len]


max_transitions = int(vocab_size*sparsity)

tgrams = ['GT', 'Uniform', 'Unigram', 'Bigram', 'Trigram', 
                'Tetragram', 'Quadragram', 'Pentragram', 'Hexagram',
                'Heptagram', 'Octagram'] 

tgram_string = tgrams[0:t_gram_list[-1]+1]

log_files = open_log_files(order_list=test_order_list, save_path=save_path)

model_seed = 0
torch.manual_seed(model_seed)

config = GPT.get_default_config()
config.model_type = None  # We're not using a pre-defined model type
config.n_layer = num_layers
config.n_head = num_heads
config.n_embd = dim
config.vocab_size = vocab_size  # Replace with your actual vocabulary size
config.block_size = max_order*sequence_length # Max possible seq. length the model can take
config.if_layer_norm = if_layer_norm
config.if_mlp = if_mlp

#Create the mode
transformer = GPT(config)
transformer = transformer.to(device)

train_transformer_gpt(transformer, steps=steps, vocab_size = vocab_size, 
                        max_transitions = max_transitions,
                        log_interval = log_interval, save_interval = save_interval,
                        sequence_length = sequence_length, 
                        test_seq_len = test_seq_len,
                        lr = lr, batch_size = bs, 
                        order_list = order_list, 
                        test_order_list=test_order_list, 
                        t_gram_list = t_gram_list, 
                        device = device, tgram_string=tgram_string, 
                        log_files=log_files, save_path=save_path)
