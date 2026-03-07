import torch
import torch.nn as nn
from .base import Module

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Small weights (0.01) keep inputs to sigmoid/tanh in the active (steep) region
        # where gradients are highest, preventing "saturation" (vanishing gradients).
        self.W_ih = nn.Parameter(torch.randn(input_size, 4 * hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, state):
        h, c = state
        
        # Compute all gate inputs in one batch
        gates = x @ self.W_ih + h @ self.W_hh + self.b
        i, f, c_tilde, o = gates.chunk(4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_tilde = torch.tanh(c_tilde)
        o = torch.sigmoid(o)
        
        # Additive cell state update
        c_next = f * c + i * c_tilde
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
