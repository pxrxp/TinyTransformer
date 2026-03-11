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
