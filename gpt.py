"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from minigpt_utils import CfgNode as CN
from transformers import GPT2Config, GPT2Model
 

def create_padding_mask(padded_sequences, padding_idx):
    # Assume padding_value is 0
    return (padded_sequences == padding_idx)  # Shape: [batch_size, 1, 1, max_seq_length]

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    # def __init__(self, config,n_head):
    #     super().__init__()
    #     assert config.n_embd % n_head == 0
    #     # key, query, value projections for all heads, but in a batch
    #     self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    #     # output projection
    #     self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    #     # regularization
    #     self.attn_dropout = nn.Dropout(config.attn_pdrop)
    #     self.resid_dropout = nn.Dropout(config.resid_pdrop)
    #     # causal mask to ensure that attention is only applied to the left in the input sequence
    #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
    #                                  .view(1, 1, config.block_size, config.block_size))
    #     self.n_head = n_head
    #     self.n_embd = config.n_embd

    # def forward(self, x):
    #     B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

    #     # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    #     q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
    #     k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    #     q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    #     v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    #     # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    #     att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #     att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    #     att = F.softmax(att, dim=-1)
    #     att = self.attn_dropout(att)
    #     y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    #     y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    #     # output projection
    #     y = self.resid_dropout(self.c_proj(y))
    #     return y

    def __init__(self, config,n_head):
        super().__init__()
        assert config.n_embd % n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = n_head
        self.n_embd = config.n_embd

        # relative positional embeddings
        self.embk = nn.Parameter(0.02 * torch.randn(config.block_size, config.n_embd // self.n_head))
        self.embv = nn.Parameter(0.02 * torch.randn(config.block_size, config.n_embd // self.n_head))

    def forward(self, x, pad_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # relative positional embeddings
        q2 = q.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, C // self.n_head)
        relative_pos = torch.arange(T)[:, None] - torch.arange(T)[None, :]
        relative_pos = relative_pos.tril().int().to(x.device)
        rk = self.embk[relative_pos]
        rv = self.embv[relative_pos]

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att1 = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att2 = q2 @ rk.transpose(1, 2)
        att2 = att2.transpose(0, 1).contiguous().view(B, self.n_head, T, T)
        att = att1 + att2
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # Apply padding mask
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            att = att.masked_fill(pad_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y1 = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y2 = att.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, T)
        y2 = y2 @ rv
        y2 = y2.transpose(0, 1).contiguous().view(B, self.n_head, T, C // self.n_head)
        y = y1 + y2
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def extract_attention_scores(self, x, pad_mask=None):
        """
        Extracts and returns the attention scores (after softmax) for the input.
        """
        B, T, C = x.size()
        q, k, _ = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # relative positional embeddings
        q2 = q.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, C // self.n_head)
        relative_pos = torch.arange(T)[:, None] - torch.arange(T)[None, :]
        relative_pos = relative_pos.tril().int().to(x.device)
        rk = self.embk[relative_pos]

        att1 = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att2 = q2 @ rk.transpose(1, 2)
        att2 = att2.transpose(0, 1).contiguous().view(B, self.n_head, T, T)
        att = att1 + att2
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(pad_mask, float('-inf'))
        
        att_scores = F.softmax(att, dim=-1)
        return att_scores

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config,n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config,n_head)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

        self.if_layer_norm = config.if_layer_norm
        self.if_mlp= config.if_mlp

    def forward(self, x, pad_mask):
        #x = x + self.attn(self.ln_1(x),pad_mask)
        #x = x + self.mlpf(self.ln_2(x))

        x = x + self.attn(self.ln_1(x) if self.if_layer_norm else x, pad_mask)
        y = self.ln_2(x) if self.if_layer_norm else x
        y = self.mlpf(y) if self.if_mlp else y

        if self.if_layer_norm or self.if_mlp:
            x = x + y
    
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1

        C.if_layer_norm = None
        C.if_mlp = None
        return C

    def __init__(self, config):
        super().__init__()

        self.config = config 
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size

        self.if_layer_norm = config.if_layer_norm
        self.if_mlp = config.if_mlp

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size+1, config.n_embd, padding_idx=config.vocab_size),
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            #  h = nn.ModuleList([
            #     Block(config, n_head=config.n_head if i == 0 else 1) 
            #     for i in range(config.n_layer)
            # ]),
            h = nn.ModuleList([Block(config, config.n_head) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def extract_all_attention_scores(self, x, pad_mask=None):
        """Extract attention scores from all blocks."""
        attention_scores = {}
        x = self.transformer.wte(x)
        x = self.transformer.drop(x)
        for idx, block in enumerate(self.transformer.h):
            attn_scores = block.attn.extract_attention_scores(x, pad_mask)
            avg_attn_scores = attn_scores  # Average across the batch
            attention_scores[f"block_{idx}"] = avg_attn_scores
        return attention_scores
    
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('embk') or pn.endswith('embv'):
                    # new relative positional embeddings will NOT be weight decayed
                    no_decay.add(fpn)    

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        #print("Min index:", torch.min(idx), "Max index:", torch.max(idx))

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        #pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        #print(f"Token Embedding Shape: {tok_emb.shape}, Positional Embedding Shape: {pos_emb.shape}")
        #print(tok_emb.device, pos_emb.device)
        # Create padding mask
        pad_mask = create_padding_mask(idx, self.vocab_size).to(device)
        
        #print("pad mask details", pad_mask.shape, pad_mask.device)
        x = self.transformer.drop(tok_emb)# + pos_emb)
        for block in self.transformer.h:
            x = block(x, pad_mask)
        x = self.transformer.ln_f(x) if self.if_layer_norm else x
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.vocab_size)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
class gpt(nn.Module):
    def __init__(self, input_dim, output_dim, drop, hid_dim = 512, n_head = 8, n_layer = 6, max_position = 50):
        super().__init__()
        config = GPT2Config(vocab_size = input_dim, n_embd= hid_dim, n_layer = n_layer, n_head = n_head, \
                            activation_function= 'gelu', n_positions= max_position, \
                             resid_pdrop = drop, embd_pdrop = drop, attn_pdrop = drop, use_cache=False, n_inner = 1)
        
        self.GPT2= GPT2Model(config)
        self.lin2= nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        hidden = self.GPT2(input_ids= x, attention_mask = torch.ones_like(x)).last_hidden_state
        last= self.lin2(hidden)

        return last   
