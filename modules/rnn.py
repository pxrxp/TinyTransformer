import torch
import torch.nn as nn
from .base import Module

class RNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        # x: (batch_size, input_size)
        # h: (batch_size, hidden_size)
        h_next = torch.tanh(x @ self.W_ih + h @ self.W_hh + self.b)
        return h_next
