import torch
import torch.nn as nn
import math
from .base import Module
from .attention import MultiHeadAttention
from .transformer_layers import LayerNormalization, Dropout, PositionalEncoding

class GPTBlock(Module):
    """
    A single layer of the GPT architecture.
    Unlike the original Transformer Decoder, GPT does not have an Encoder, 
    so there is no Cross-Attention. It strictly uses Masked Self-Attention 
    followed by a Feed-Forward network.
    """
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout_probability: float = 0.1):
        super().__init__()

        # Masked self-attention (causal)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(), # GPT notably uses GELU instead of ReLU
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm_self_attn = LayerNormalization(d_model)
        self.norm_feedforward = LayerNormalization(d_model)
        
        self.dropout = Dropout(dropout_probability=dropout_probability)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Masked Self-Attention
        attended_self, _ = self.self_attention(x, x, x, mask=mask)
        x = self.norm_self_attn(x + self.dropout(attended_self))

        # 2. Feed-Forward Network
        processed = self.feed_forward(x)
        x = self.norm_feedforward(x + self.dropout(processed))

        return x

class GPT(Module):
    """
    Generative Pre-trained Transformer (GPT).
    A purely autoregressive, Decoder-only architecture.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        max_len: int = 1024,
        dropout_probability: float = 0.1,
    ):
        super().__init__()
        
        # Token Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # In GPT, Positional Encodings are typically learned embeddings, 
        # but we use standard sinusoidal here for educational continuity.
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout_probability)
        
        # Stack of GPT Blocks (Decoder-only, no cross-attention)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, dim_feedforward, dropout_probability)
            for _ in range(num_layers)
        ])
        
        # Final Unembedding Layer
        self.final_linear = nn.Linear(d_model, vocab_size, bias=False)

    def generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            seq_len = input_ids.size(1)
            mask = self.generate_causal_mask(seq_len).to(input_ids.device)
            
        # Scale embeddings by sqrt(d_model) before adding positional encoding
        # This prevents the unit-variance positional signals from drowning out the token meanings.
        x = self.token_embedding(input_ids) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.positional_encoding(x)
        
        for block in self.blocks:
            x = block(x, mask)
            
        logits = self.final_linear(x)
        return logits
