import torch, random, numpy as np, time
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
