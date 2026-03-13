import torch
import torch.nn as nn
import math
from .base import Module
from .attention import MultiHeadAttention


class LayerNormalization(Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # nn.Parameter: marks tensors as trainable model weights. 
        # These appear in model.parameters() and are updated by the optimizer.
        # shape: (d_model)
        self.gamma = nn.Parameter(torch.ones(d_model))
        # shape: (d_model)
        self.beta = nn.Parameter(torch.zeros(d_model))

        # small constant added to variance to avoid division by zero
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x: (batch, seq, d_model)

        # x.mean(dim=-1): calculates average across the last (d_model) dimension.
        # keepdim=True: preserves the dimension as size 1 instead of removing it.
        # Consequence: shape remains (batch, seq, 1) so it can broadcast-subtract from x (batch, seq, d_model).
        mean = x.mean(dim=-1, keepdim=True)
        
        # unbiased=False: calculates population variance (division by N) instead of sample (N-1).
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Element-wise normalization: (input - mean) / sqrt(variance + epsilon)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # Shape of x_normalized: (batch, seq, d_model)
        # Shape of self.gamma: (d_model)
        # Shape of self.beta: (d_model)
        # Broadcasting: self.gamma and self.beta are broadcasted to match the shape of x_normalized.

        # Example
        # gamma = [g1 g2 g3 g4]

        # becomes
        # [[g1 g2 g3 g4],
        #  [g1 g2 g3 g4],
        #  [g1 g2 g3 g4]]

        return self.gamma * x_normalized + self.beta


class Dropout(Module):
    def __init__(self, dropout_probability: float = 0.1):
        super().__init__()
        self.dropout_probability = dropout_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.training: boolean inherited from nn.Module. False during model.eval().
        if not self.training or self.dropout_probability == 0:
            return x

        # Only during training, randomly drop some neurons to prevent overfitting.
        # During evaluation dropout is disabled.

        # torch.rand_like(x): generates random uniform [0, 1) values with same shape/device/dtype as x.
        # (rand > p): returns a boolean mask. .float() converts True->1.0 and False->0.0.
        mask = (torch.rand_like(x) > self.dropout_probability).float()


        # Example
        # x shape: (2,3,4)

        # mask example:
        # [[1 0 1 1],
        #  [0 1 1 0],
        #  [1 1 0 1]]
        
        # Scale remaining elements by 1/(1-p)
        # Consequence: the expected value of the outputs remains consistent between training and inference.
        return x * mask / (1.0 - self.dropout_probability)


class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout_probability: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout_probability=dropout_probability)

        # torch.zeros: initializes a 2D buffer on the CPU.
        positional_encodings = torch.zeros(max_len, d_model)

        # torch.arange(start=0, end=max_len): returns a 1D tensor [0, 1, 2, ..., max_len-1].
        # .unsqueeze(1): inserts a new dimension at index 1.
        # shape: (max_len,) -> (max_len, 1)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # torch.arange(0, d_model, 2) -> [0, 2, 4, ...]
        # divisors: calculates 10000^(-i / d_model) for i in [0, 2, 4, ...]
        # computes: 10000^(-2i / d_model)
        # resulting tensor shape: (d_model/2,)
        divisors = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # positions shape: (max_len, 1)
        # divisors shape:  (d_model/2,)
        # broadcasting produces: (max_len, d_model/2)
        
        # All rows should have sin at even columns and cos at odd columns.
        positional_encodings[:, 0::2] = torch.sin(positions * divisors)
        positional_encodings[:, 1::2] = torch.cos(positions * divisors)

        # Example when d_model = 4
        # arange(0,4,2) -> [0,2]
        # divisors = [10000^(0/4), 10000^(-2/4)] = [1, 0.01]

        # shapes
        # positions (3,1)
        # divisors  (2)

        # positions:
        # [[0],
        #  [1],
        #  [2]]

        # divisors acts like
        # [[1,    0.01],
        #  [1,    0.01],
        #  [1,    0.01]]

        # positions * divisors (broadcasted):
        # [[0,    0],
        #  [1,    0.01],
        #  [2,    0.02]]

        # resulting positional encodings:
        # pos 0 → [sin(0), cos(0), sin(0),     cos(0)]
        # pos 1 → [sin(1), cos(1), sin(0.01),  cos(0.01)]
        # pos 2 → [sin(2), cos(2), sin(0.02),  cos(0.02)]

        # .unsqueeze(0): adds a batch dimension
        # shape: (max_len, d_model) -> (1, max_len, d_model)
        positional_encodings = positional_encodings.unsqueeze(0)

        # register_buffer: adds a non-trainable tensor to state_dict.
        # It will be moved to GPU automatically when model.to('cuda') is called.
        self.register_buffer('positional_encodings', positional_encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # Example:
        # batch = 2 sequences, seq_len = 3 tokens, d_model = 4 embedding size
        # Tokens:
        # seq 0: ["dog", "bit", "human"]
        # seq 1: ["cat", "ate", "fish"]
        #
        # Example embedding values (simplified):
        # x[0] = [[0.1, 0.2, 0.3, 0.4],   # "dog"
        #          [0.5, 0.6, 0.7, 0.8],   # "bit"
        #          [0.9, 1.0, 1.1, 1.2]]   # "human"
        # x[1] = [[0.2, 0.1, 0.4, 0.3],   # "cat"
        #          [0.6, 0.5, 0.8, 0.7],   # "ate"
        #          [1.0, 0.9, 1.2, 1.1]]   # "fish"

        # Slice positional encodings to match sequence length
        pos = self.positional_encodings[:, :x.size(1)]  # shape: (1, seq_len, d_model)
        # Example positional values (simplified):
        # pos = [[[0.01, 0.02, 0.03, 0.04],  # position 0
        #         [0.05, 0.06, 0.07, 0.08],  # position 1
        #         [0.09, 0.10, 0.11, 0.12]]] # position 2

        # Add positional encodings to embeddings
        # x + pos, automatically repeated across batch
        # Resulting embeddings:
        # seq 0:
        # [[0.11, 0.22, 0.33, 0.44],  # "dog" + pos0
        #  [0.55, 0.66, 0.77, 0.88],  # "bit" + pos1
        #  [0.99, 1.10, 1.21, 1.32]]  # "human" + pos2
        # seq 1:
        # [[0.21, 0.12, 0.43, 0.34],  # "cat" + pos0
        #  [0.65, 0.56, 0.87, 0.78],  # "ate" + pos1
        #  [1.09, 1.00, 1.31, 1.22]]  # "fish" + pos2
        x = x + pos

        # Apply dropout
        return self.dropout(x)


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