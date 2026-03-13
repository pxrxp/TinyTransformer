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
        """
        x: (batch_size, input_size) - input at the current time step
        state: (h, c) - hidden and cell states from the previous time step
               h: (batch_size, hidden_size)
               c: (batch_size, hidden_size)
        """
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

class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, state0=None):
        """
        x: (seq_len, batch_size, input_size) - input sequence
        state0: (h0, c0) - optional initial hidden and cell states
        """
        # x: (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.size()
        
        if state0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
            state = (h0, c0)
        else:
            state = state0
        
        h_all = []
        for i in range(seq_len):
            state = self.cell(x[i], state)
            h_all.append(state[0])
            
        return torch.stack(h_all), state
