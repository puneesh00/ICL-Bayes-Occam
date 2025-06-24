import torch
import torch.nn as nn
from model_utils import MultiheadAttentionWithRPE



class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length=10000):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        #Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def create_positional_encoding(self, max_seq_length, embed_dim):
        pos_encoding = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding
    def forward(self, x, src_mask=None):

        device = x.device
        seq_length = x.shape[1]
        #print(x.shape)
        x = self.embedding(x)
    
        x = x + self.positional_encoding[:, :seq_length, :].to(device)
    
        for layer in self.transformer_encoder.layers:
            x = layer(x, src_mask=src_mask)
    
        x = x[:, :, :]  # Select everything except the last
        x = self.fc_out(x)
        return x
    
class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, if_layer_norm, if_dropout, rpe, dropout=0.1):
        super(CustomTransformerBlock, self).__init__()
        if rpe:
            self.self_attention = MultiheadAttentionWithRPE(embed_dim, num_heads, if_dropout, dropout=dropout)
        else:
            self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        #self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        #self.if_layer_norm = if_layer_norm
        #self.if_dropout = if_dropout

        self.rpe = rpe
    def forward(self, x, mask=None):


        # Self-attention
        norm_x = self.layer_norm1(x)
        attn_output = self.self_attention(norm_x, mask=mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)

        # # Self-attention
        # if self.rpe:
        #     attn_output = self.self_attention_rpe(x, mask)
        # else:
        #     attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)

        # if self.if_dropout:
        #     attn_output = self.dropout(attn_output)    
        
        # x = x + attn_output
        # if self.if_layer_norm:
        #     x = self.layer_norm(x)
       
        
        # # Feed-forward
        # ff_output = self.linear(x)    
        
        # if self.if_dropout:
        #     ff_output = self.dropout(ff_output)

        # x = x + self.dropout(ff_output)
        
        # if self.if_layer_norm:
        #     x = self.layer_norm(x)
        
        return x

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, if_layer_norm, if_dropout, rpe, max_seq_length=10000):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(embed_dim, num_heads, if_layer_norm, if_dropout, rpe) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.rpe = rpe
    
    def create_positional_encoding(self, max_seq_length, embed_dim):
        pos_encoding = torch.zeros(max_seq_length, embed_dim)
        for pos in range(max_seq_length):
            for i in range(0, embed_dim, 2):
                angle = pos / (10000 ** (2 * i / embed_dim))
                pos_encoding[pos, i] = torch.sin(torch.tensor(angle))
                if i + 1 < embed_dim:
                    angle_next = pos / (10000 ** (2 * (i + 1) / embed_dim))
                    pos_encoding[pos, i + 1] = torch.cos(torch.tensor(angle_next))
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding
    
    def forward(self, x, mask=None):
        device = x.device
        seq_length = x.shape[1]
        x = self.embedding(x)
       
        if not self.rpe:
            x = x + self.positional_encoding[:, :seq_length, :].to(device)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        #x = x[:, :, :]  # Use the last token's embedding
        x = self.fc_out(x)
        return x
