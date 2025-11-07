from configurator import Config
from tokenizer import CharTokenizer
from dataset import TextDataset
from dataloader import create_dataloader
from optimizer.adamw import create_optimizer
from engine import train_one_epoch
from sentiment_dataset import SentimentDataset
from translation_dataset import TranslationDataset
import torch
import torch.nn as nn
from checkpoint_manager import save_checkpoint

from model.decoder import Decoder
from model.encoder import Encoder
from model.translation import TranslationModel
from model.transformer import GPT


def train_model(model, dataset, dataloader, optimizer, criterion, checkpoint_path, epochs, checkpoint_every, device):
    model = model.to(device)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | loss: {loss:.4f}")
        
        # Sauvegarder si meilleure loss
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, optimizer, epoch, path=checkpoint_path)
            print(f"  ✓ Meilleure loss sauvegardée: {loss:.4f}")
        
        # Sauvegarde périodique
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, path=checkpoint_path.replace('.pt', f'_epoch_{epoch+1}.pt'))
    
    print(f"\n✅ Entraînement terminé. Meilleure loss: {best_loss:.4f}")

if __name__ == "__main__":
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Décodeur : génération de texte (GPT-like sur textgen_data)
    print('\n--- Entraînement du modèle: Decoder (génération de texte) ---')
    text = open("data/textgen_data.txt", "r", encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = tok.encode(text)
    train_data = data[:int(0.9*len(data))]
    train_ds = TextDataset(train_data, cfg.block_size)
    train_dl = create_dataloader(train_ds, cfg.batch_size)
    model = GPT(tok.vocab_size, cfg.block_size, cfg.embed_dim, cfg.num_layers, cfg.num_heads, cfg.ff_hidden_dim, cfg.dropout)
    optimizer = create_optimizer(model, cfg.lr, getattr(cfg, 'weight_decay', 0.01))
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_ds, train_dl, optimizer, criterion, "checkpoints/decoder_ckpt.pt", cfg.epochs, cfg.checkpoint_every, device)

    # 2. Encodeur : classification de sentiment
    print('\n--- Entraînement du modèle: Encoder (classification de sentiment) ---')
    text_sent = " ".join([l.split("\t",1)[1] for l in open("data/sentiment_data.txt", encoding="utf-8") if "\t" in l])
    tokenizer_sent = CharTokenizer(text_sent)
    sent_ds = SentimentDataset("data/sentiment_data.txt", tokenizer_sent, block_size=cfg.block_size)
    sent_dl = create_dataloader(sent_ds, cfg.batch_size)
    # Encoder amélioré avec plus de paramètres
    model = Encoder(
        vocab_size=tokenizer_sent.vocab_size, 
        embed_dim=cfg.embed_dim, 
        num_classes=3,
        num_layers=2,
        hidden_dim=cfg.ff_hidden_dim,
        dropout=cfg.dropout
    )
    optimizer = create_optimizer(model, cfg.lr, getattr(cfg, 'weight_decay', 0.01))
    criterion = nn.CrossEntropyLoss()
    train_model(model, sent_ds, sent_dl, optimizer, criterion, "checkpoints/encoder_ckpt.pt", cfg.epochs, cfg.checkpoint_every, device)

    # 3. Traduction EN-FR réel
    print('\n--- Entraînement du modèle: TranslationModel (traduction EN<>FR) ---')
    # Construire deux tokenizers : source=anglais, cible=français
    text_src = " ".join([l.split("\t",1)[0] for l in open("data/translation_data.txt", encoding="utf-8") if "\t" in l])
    text_tgt = " ".join([l.split("\t",1)[1] for l in open("data/translation_data.txt", encoding="utf-8") if "\t" in l])
    tokenizer_src = CharTokenizer(text_src)
    tokenizer_tgt = CharTokenizer(text_tgt)
    trans_ds = TranslationDataset("data/translation_data.txt", tokenizer_src, tokenizer_tgt, block_size=cfg.block_size)
    trans_dl = create_dataloader(trans_ds, cfg.batch_size)
    # TranslationModel amélioré
    model = TranslationModel(
        vocab_src=tokenizer_src.vocab_size, 
        vocab_tgt=tokenizer_tgt.vocab_size, 
        embed_dim=cfg.embed_dim, 
        block_size=cfg.block_size,
        ff_hidden_dim=cfg.ff_hidden_dim,  # ✅ Changé de hidden_dim à ff_hidden_dim
        dropout=cfg.dropout
    )
    optimizer = create_optimizer(model, cfg.lr, getattr(cfg, 'weight_decay', 0.01))
    criterion = nn.CrossEntropyLoss()
    train_model(model, trans_ds, trans_dl, optimizer, criterion, "checkpoints/translation_ckpt.pt", cfg.epochs, cfg.checkpoint_every, device)