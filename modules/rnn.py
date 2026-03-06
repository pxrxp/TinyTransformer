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

class RNN(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        # x: (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        h = h0
        outputs = []
        for i in range(seq_len):
            h = self.cell(x[i], h)
            outputs.append(h)
        
        return torch.stack(outputs), h
