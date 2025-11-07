import torch
def create_optimizer(model, lr, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
