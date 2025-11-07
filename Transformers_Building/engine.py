import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # 🔧 Déballer la sortie du modèle
        logits, loss = model(x, y)
        
        # GPT renvoie déjà une loss calculée, sinon on la calcule manuellement
        if loss is None:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
