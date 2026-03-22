import torch
import torch.nn as nn
import math
from .base import Module
from .attention import MultiHeadAttention
from .transformer_layers import LayerNormalization, Dropout, PositionalEncoding

class TransformerDecoderBlock(Module):
    """
    A single layer of the Transformer Decoder.
    It takes the target sequence embeddings (what we've generated so far) and the 
    output from the Encoder (the context), and refines the target embeddings 
    through masked self-attention, cross-attention, and a feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout_probability: float = 0.1):
        super().__init__()

        # Self-attention for the decoder (masked so it can't see the future)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Cross-attention connecting target words to source concepts
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm_self_attn = LayerNormalization(d_model)
        self.norm_cross_attn = LayerNormalization(d_model)
        self.norm_feedforward = LayerNormalization(d_model)
        
        self.dropout = Dropout(dropout_probability=dropout_probability)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor = None, target_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, target_seq_len, d_model) - Target sequence embeddings (what we are translating to)
        encoder_output: (batch, source_seq_len, d_model) - The contextual "thought vectors" from the Encoder
        source_mask: Optional mask for source padding
        target_mask: Causal mask to prevent looking at future target words
        """

        # 1. Masked Self-Attention (prevents looking ahead at future target tokens)
        attended_self, _ = self.self_attention(x, x, x, mask=target_mask)
        x = self.norm_self_attn(x + self.dropout(attended_self))

        # 2. Cross-Attention (Query = Decoder, Key/Value = Encoder)
        attended_cross, _ = self.cross_attention(q=x, k=encoder_output, v=encoder_output, mask=source_mask)
        x = self.norm_cross_attn(x + self.dropout(attended_cross))

        # 3. Feedforward Network
        # Apply a position-wise feedforward network (same MLP applied independently to each token).
        processed = self.feed_forward(x)
        x = self.norm_feedforward(x + self.dropout(processed))
        # Final output: refined embeddings ready for the next block or vocabulary prediction.

        return x

class TransformerDecoder(Module):
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
        
        # Maps target token IDs to dense vectors of size d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Scale embeddings by sqrt(d_model)
        self.embedding_scale = math.sqrt(d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout_probability)
        
        # Stack no. of identical sequential decoder blocks using nn.ModuleList
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, dim_feedforward, dropout_probability)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor = None, target_mask: torch.Tensor = None) -> torch.Tensor:
        """
        token_ids: (batch, target_seq_len) - Target sequence word IDs.
        Output: (batch, target_seq_len, d_model) - Refined target sequence embeddings.
        """
        x = self.token_embedding(token_ids) * self.embedding_scale        
        # Injects the order of the target words.
        x = self.positional_encoding(x)
        
        # Pass the representation through each block, refining the prediction context 
        # using both its own past and the encoder's output.
        for block in self.decoder_blocks:
            x = block(x, encoder_output, source_mask, target_mask)
            
        return x
