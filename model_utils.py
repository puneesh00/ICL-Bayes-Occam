import torch
import torch.nn as nn
import numpy as np


class RelativePositionalEncoding(nn.Module):
    def __init__(self, num_heads, max_seq_length, embed_dim):
        super(RelativePositionalEncoding, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Create a learnable parameter for relative position encodings
        self.relative_positional_encoding = nn.Parameter(torch.randn(num_heads, max_seq_length * 2 - 1, embed_dim // num_heads) * 0.02)

    def forward(self, seq_length):
        # Compute the relative positions for each token pair
        pos_indices = torch.arange(seq_length).unsqueeze(1) - torch.arange(seq_length).unsqueeze(0)
        pos_indices += (seq_length - 1)  # Shift to make indices non-negative
        return self.relative_positional_encoding[:, pos_indices]
    
class MultiheadAttentionWithRPE(nn.Module):
    def __init__(self, embed_dim, num_heads, if_dropout, dropout=0.1, max_seq_length=5000):
        super(MultiheadAttentionWithRPE, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        self.if_dropout = if_dropout
        # Linear layers to project input to queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Linear layer to project concatenated outputs from all heads back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Relative positional encoding module
        self.rpe = RelativePositionalEncoding(num_heads, max_seq_length, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # Project inputs to queries, keys, and values
        queries = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the dot product attention scores (without RPE)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Add relative positional encoding to the attention scores
        rpe = self.rpe(seq_length)  # [num_heads, seq_length, seq_length, head_dim]
        rpe_scores = torch.einsum('bhqd,hqkd->bhqk', queries, rpe)
        attn_scores += rpe_scores

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Compute the attention weights using softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        #if self.if_dropout:
        attn_weights = self.dropout(attn_weights)

        # Compute the context vectors as a weighted sum of the values
        context = torch.matmul(attn_weights, values)

        # Concatenate the context vectors from all heads and project back to the original embed_dim
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        out = self.out_proj(context)

        return out    