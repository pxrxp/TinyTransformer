import torch
import torch.nn as nn
from .base import Module
from .transformer_encoder import TransformerEncoder

class BERT(Module):
    """
    Bidirectional Encoder Representations from Transformers (BERT).
    A purely bidirectional, Encoder-only architecture built to understand context.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        max_len: int = 512,
        dropout_probability: float = 0.1,
    ):
        super().__init__()
        
        # Core structure is identical to the Transformer's Encoder half
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            max_len=max_len,
            dropout_probability=dropout_probability
        )
        
        # Masked Language Modeling (MLM) Prediction Head
        # Projects the contextual embeddings back to the full vocabulary size
        # to guess the hidden words (the <MASK> tokens).
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        # Next Sentence Prediction (NSP) Head
        # Projects the special [CLS] token (at index 0) into 2 classes: Is_Next or Not_Next
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        BERT uses no causal mask because it is meant to look in both directions.
        It only uses a padding mask to ignore <PAD> tokens in variable-length batches.
        """
        # (batch, seq_len, d_model) Contextual representation of the entire sentence
        context_embeddings = self.encoder(input_ids, mask=padding_mask)
        
        # Predict the identity of the 15% masked words
        mlm_logits = self.mlm_head(context_embeddings)     # (batch, seq_len, vocab_size)
        
        # Extract the very first token in the sequence (The [CLS] token)
        # That token's embedding contains the aggregated meaning of the entire sequence
        cls_embedding = context_embeddings[:, 0, :]        # (batch, d_model)
        
        # Predict whether Sentence B logically follows Sentence A
        nsp_logits = self.nsp_head(cls_embedding)          # (batch, 2)
        
        return mlm_logits, nsp_logits
