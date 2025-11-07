import torch, os
def save_checkpoint(model, optimizer, epoch, path="checkpoints/ckpt.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "epoch": epoch}, path)
def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    return ckpt.get("epoch", 0)
