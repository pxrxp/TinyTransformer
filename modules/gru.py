import torch
import torch.nn as nn
from .base import Module

class GRUCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Scaling by 0.01 keeps activations in the active region (steep gradient)
        self.W_gates_ih = nn.Parameter(torch.randn(input_size, 2 * hidden_size) * 0.01)
        self.W_gates_hh = nn.Parameter(torch.randn(hidden_size, 2 * hidden_size) * 0.01)
        self.b_gates = nn.Parameter(torch.zeros(2 * hidden_size))
        
        # Candidate weights must be separate due to reset gate dependency
        self.W_cand_ih = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_cand_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_cand = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        """
        x: (batch_size, input_size) - input at the current time step
        h: (batch_size, hidden_size) - hidden state from the previous time step
        """
        gates = x @ self.W_gates_ih + h @ self.W_gates_hh + self.b_gates
        z, r = gates.chunk(2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        # Example: gates = tensor([[0.1, 0.2, 0.3, 0.4]])
        # gates.chunk(2, dim=1) -> [tensor([[0.1, 0.2]]), tensor([[0.3, 0.4]])]
        
        h_tilde = torch.tanh(x @ self.W_cand_ih + (r * h) @ self.W_cand_hh + self.b_cand)
        h_next = (1 - z) * h + z * h_tilde
        
        return h_next

class GRU(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        """
        x: (seq_len, batch_size, input_size) - input sequence
        h0: (batch_size, hidden_size) - optional initial hidden state 
        """
        # x: (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.size()
        
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h0
        
        outputs = []
        for i in range(seq_len):
            h = self.cell(x[i], h)
            outputs.append(h)
            
        return torch.stack(outputs), h
