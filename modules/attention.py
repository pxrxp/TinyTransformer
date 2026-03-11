import torch
import torch.nn as nn
import math
from .base import Module

class ScaledDotProductAttention(Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        # We have 4D tensors: (Batch, Heads, L, d_k)
        # PyTorch treats (Batch, Heads) as a 'stack' of many 2D matrices.
        # It only performs matmul/transpose on the last two dimensions (L, d_k).
        d_k = q.size(-1)
        
        # 1. Transpose: Swaps last two dims (L x d_k) -> (d_k x L)
        # 2. Matmul: (L x d_k) * (d_k x L) = (L x L) similarity matrix
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        weights = torch.softmax(scores, dim=-1)
        
        # 3. Matmul: (L x L) weights * (L x d_v) values = (L x d_v) context
        return torch.matmul(weights, v), weights

class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch, seq_len, _ = x.size()
        # view(): Breaks d_model into (Heads, d_k). Shape: (B, L, H, d_k)
        # transpose(1, 2): Swaps L and H -> (B, H, L, d_k)
        # This makes it a 'stack' of H matrices, each of size (L x d_k).
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch, heads, seq_len, d_k = x.size()
        # transpose(1, 2): Swaps back -> (B, L, H, d_k)
        # contiguous(): Re-orders memory so the H and d_k dimensions are side-by-side
        # view(): Glues (H, d_k) back into d_model -> (B, L, d_model)
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        
        context, weights = self.attention(q, k, v, mask)
        
        out = self.combine_heads(context)
        return self.w_o(out), weights
