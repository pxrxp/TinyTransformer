import torch
import torch.nn as nn
from .base import Module
from .attention import MultiHeadAttention
from .transformer_layers import LayerNormalization, Dropout


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout_probability: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm_attention = LayerNormalization(d_model)
        self.norm_feedforward = LayerNormalization(d_model)
        self.dropout = Dropout(dropout_probability=dropout_probability)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model) — input token embeddings (already including positional info)
        mask: optional mask to prevent attention to certain positions
        """

        # ===== 1. Self-Attention =====
        # Each token embedding (Q) looks at all token embeddings (K, V) to gather context.
        # Q = K = V = x for self-attention.
        # Output: new token embeddings that include information from other tokens
        attended, _ = self.attention(x, x, x, mask)
        # attended shape: (batch, seq_len, d_model)

        # ===== 2. Residual + LayerNorm =====
        # Add original embeddings (residual connection) to attended output.
        # Normalize to stabilize training.
        x = self.norm_attention(x + self.dropout(attended))
        # x shape remains (batch, seq_len, d_model)

        # ===== 3. Feedforward =====
        # Each token embedding is independently processed through a small MLP
        # to add non-linear transformation.
        processed = self.feed_forward(x)
        # processed shape: (batch, seq_len, d_model)

        # ===== 4. Residual + LayerNorm =====
        # Add the input to the feedforward network back (residual)
        # Normalize the result.
        x = self.norm_feedforward(x + self.dropout(processed))
        # Final output: updated token embeddings including context and nonlinear transformation

        return x
