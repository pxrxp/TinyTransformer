import torch
import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: str):
        """
        Saves the model weights to a file.
        path: (str) - path where the weights will be saved
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Loads the model weights from a file.
        path: (str) - path to the weights file
        """
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
