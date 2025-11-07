# model/normalization.py
import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    """
    Simple wrapper for PyTorch LayerNorm with optional epsilon and bias control.
    """
    def __init__(self, embed_dim, eps=1e-5, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return self.weight * normed + self.bias
        return self.weight * normed
    