import torch
import torch.nn as nn
from .base import Module
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

class Transformer(Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        max_len: int = 5000,
        dropout_probability: float = 0.1,
    ):
        super().__init__()

        """
        source_vocab_size: The number of unique words in the source language (e.g., English).
        target_vocab_size: The number of unique words in the target language (e.g., Spanish).
        d_model: The dimension of the model's internal representations (e.g., 512).
        num_heads: The number of attention heads (e.g., 8).
        num_encoder_layers: The number of encoder layers (e.g., 6).
        num_decoder_layers: The number of decoder layers (e.g., 6).
        dim_feedforward: The dimension of the feedforward network (e.g., 2048).
        max_len: The maximum length of the input sequences (e.g., 5000).
        dropout_probability: The dropout probability (e.g., 0.1).
        """
        
        self.encoder = TransformerEncoder(
            vocab_size=source_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers,
            max_len=max_len,
            dropout_probability=dropout_probability
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=target_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_decoder_layers,
            max_len=max_len,
            dropout_probability=dropout_probability
        )
        
        # Final linear layer: Maps the complex d_model concept vector 
        # into scores (logits) for every single word in the target vocabulary.
        self.final_linear = nn.Linear(d_model, target_vocab_size)
        
        # Final Softmax layer: explicitly added to match the architecture diagram.
        # It turns the raw scores from the linear layer into probabilities (0.0 to 1.0).
        self.softmax = nn.Softmax(dim=-1)

    def generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Creates a "lower-triangular" matrix of 1s to prevent attending to the future.
        
        Example for seq_len = 3:
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
         
        When this is passed to attention, the 0s become -infinity, meaning:
        - Word 1 can only look at Word 1.
        - Word 2 can look at Word 1 and Word 2.
        - Word 3 can look at Word 1, 2, and 3.
        """
        # torch.ones(seq_len, seq_len): creates a square matrix of 1s
        # torch.tril: extracts lower triangular part (keeps 1s on/below diagonal, zeros out above)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # Add batch and head dimensions for broadcasting: (1, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, source: torch.Tensor, target: torch.Tensor, source_mask: torch.Tensor = None, target_mask: torch.Tensor = None) -> torch.Tensor:
        """
        source: (batch, source_seq_len) - Input token IDs. Ex: ["I", "am", "happy"]
        target: (batch, target_seq_len) - Output token IDs, but "shifted right by one position". Ex: ["<SOS>", "Yo", "soy", "feliz"]
        """
        
        # If no target mask is explicitly provided, we build the causal mask automatically.
        if target_mask is None:
            target_seq_len = target.size(1)
            # Send mask to the same device as the target tensor (e.g., CPU or CUDA)
            target_mask = self.generate_causal_mask(target_seq_len).to(target.device)
            
        # ===== 1. Encode Source =====
        # The entire source sentence is processed in parallel to generate contextual "thought vectors".
        # Shape: (batch, source_seq_len, d_model)
        encoder_output = self.encoder(source, mask=source_mask)
        
        # ===== 2. Decode Target =====
        # The shifted target sequence is processed. For each word, it looks at the words before it (via self-attention + target_mask)
        # and looks at the encoder concepts (via cross-attention).
        # Shape: (batch, target_seq_len, d_model)
        decoder_output = self.decoder(target, encoder_output, source_mask, target_mask)
        
        # ===== 3. Final Linear & Softmax =====
        # Linear converts the d_model features to vocabulary-sized predictions.
        # Shape: (batch, target_seq_len, target_vocab_size)
        logits = self.final_linear(decoder_output)
        
        # Softmax turns the logits into actual probabilities as shown in the diagram.
        # Shape remains: (batch, target_seq_len, target_vocab_size)
        probabilities = self.softmax(logits)
        
        return probabilities
