# model/positional_encoding.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding used for GPT-like models.
    Adds positional information to the token embeddings.
    """
    def __init__(self, embed_dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encodings added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
