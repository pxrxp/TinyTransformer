import torch
import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
