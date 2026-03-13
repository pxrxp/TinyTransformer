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