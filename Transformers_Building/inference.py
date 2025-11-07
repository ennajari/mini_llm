import torch
from model.transformer import GPT
from tokenizer import CharTokenizer
from configurator import Config
import os

# ============================================================
# Charger la configuration
# ============================================================
cfg = Config("config.yaml")

# Charger le texte brut pour initialiser le tokenizer
with open(cfg.data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Initialiser le tokenizer (char-level)
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

# Choisir le device (CPU only si pas de GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# Charger le modèle GPT
# ============================================================
model = GPT(
    vocab_size=vocab_size,
    block_size=cfg.block_size,
    embed_dim=cfg.embed_dim,
    num_layers=cfg.num_layers,
    num_heads=cfg.num_heads,
    ff_hidden_dim=cfg.ff_hidden_dim,
    dropout=cfg.dropout
).to(device)

# ============================================================
# Charger le checkpoint sauvegardé
# ============================================================
checkpoint_path = "checkpoints/ckpt.pt"
assert os.path.exists(checkpoint_path), "❌ ERREUR: checkpoint introuvable !"

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("✅ Modèle + checkpoint chargé avec succès")

# ============================================================
# Fonction de génération de texte
# ============================================================
@torch.no_grad()
def generate(model, start_tokens, max_new_tokens, tokenizer, device):
    model.eval()
    idx = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        # Fenêtre contextuelle limitée à block_size
        idx_cond = idx[:, -cfg.block_size:]

        # 🔧 Le modèle retourne (logits, loss), donc on déballe le tuple
        output = model(idx_cond)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # On ne garde que les logits du dernier token
        logits = logits[:, -1, :]

        # On calcule les probabilités avec softmax
        probs = torch.softmax(logits, dim=-1)

        # Échantillonnage stochastique du prochain token
        idx_next = torch.multinomial(probs, num_samples=1)

        # On concatène le token généré à la séquence
        idx = torch.cat((idx, idx_next), dim=1)

    # Décodage en texte
    return tokenizer.decode(idx[0].tolist())

# ============================================================
# Exemple de génération
# ============================================================
prompt = "hello"
start_tokens = tokenizer.encode(prompt)

output = generate(
    model=model,
    start_tokens=start_tokens,
    max_new_tokens=100,   # longueur de texte à générer
    tokenizer=tokenizer,
    device=device
)

print("\n===== OUTPUT =====")
print(output)
