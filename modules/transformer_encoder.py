import torch
import torch.nn as nn
import math
from .base import Module
from .transformer_layers import PositionalEncoding
from .transformer_block import TransformerBlock


class TransformerEncoder(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        max_len: int = 5000,
        dropout_probability: float = 0.1,
    ):
        super().__init__()
        # nn.Embedding: initializes a lookup table of trainable (vocab_size, d_model) vectors.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # math.sqrt(d_model): constant scaling factor.
        # Embedding weights are initialized with small variance; scaling them up 
        # prevents them from being overpowered by the unit-variance positional encoding.
        self.embedding_scale = math.sqrt(d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout_probability)
        
        # nn.ModuleList: registers a list of modules as sub-layers of the model.
        # Unlike a native Python list, ModuleList ensures parameters are seen by .parameters().
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward, dropout_probability)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) — integer IDs of tokens in the vocabulary
        mask: optional mask to prevent attention to padding or future tokens
        """

        # ===== 1. Convert token IDs to embeddings =====
        # Each token ID is mapped to a trainable vector of size d_model
        # This is what the model "understands" as the token meaning
        x = self.token_embedding(token_ids) * self.embedding_scale
        # x shape: (batch, seq_len, d_model)

        # ===== 2. Add positional encoding =====
        # Injects information about the position of each token in the sequence
        # So the model knows the order of words
        x = self.positional_encoding(x)
        # x shape: (batch, seq_len, d_model)

        # ===== 3. Pass through each Transformer block =====
        # Each block updates token embeddings to include context from all other tokens
        for block in self.encoder_blocks:
            x = block(x, mask)
            # After each block: token embeddings are more contextualized

        # ===== 4. Return final contextual embeddings =====
        # x now contains token representations that include:
        # - original token meaning
        # - positional information
        # - context from surrounding tokens through attention
        # Shape: (batch, seq_len, d_model)
        return x
