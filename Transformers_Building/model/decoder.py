# model/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward
from .positional_encoding import PositionalEncoding
from .normalization import LayerNorm


class TransformerDecoderBlock(nn.Module):
    """Bloc decoder transformer avec self-attention causale"""
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN architecture avec masque causal (déjà dans MultiHeadSelfAttention)
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class Decoder(nn.Module):
    """
    Decoder Transformer pour génération de texte autoregressive.
    Compatible avec l'architecture GPT existante.
    """
    def __init__(self, vocab_size=128, block_size=512, embed_dim=128, 
                 num_layers=6, num_heads=4, ff_hidden_dim=512, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, block_size)
        
        # Stack de blocs transformer decoder
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_ln = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Tête de langage
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Partage des poids entre embedding et lm_head (optionnel mais efficace)
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialisation des poids
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, y=None, max_length=None):
        """
        Args:
            x: (batch, seq_len) indices de tokens
            y: (batch, seq_len) targets pour calcul de perte (optionnel)
            max_length: ignoré (pour compatibilité API)
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalaire si y fourni, sinon None
        """
        B, T = x.size()
        
        if T > self.block_size:
            raise ValueError(f"Séquence de longueur {T} dépasse block_size {self.block_size}")
        
        # Embeddings + encodage positionnel
        tok_emb = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        emb = self.pos_encoding(tok_emb)
        emb = self.dropout(emb)
        
        # Passer par les blocs transformer
        for block in self.blocks:
            emb = block(emb)
        
        emb = self.final_ln(emb)
        
        # Projection vers vocabulaire
        logits = self.lm_head(emb)  # (batch, seq_len, vocab_size)
        
        loss = None
        if y is not None:
            # Calcul de la perte sur toute la séquence
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Génération autoregressive de tokens.
        
        Args:
            idx: (batch, seq_len) contexte initial
            max_new_tokens: nombre de tokens à générer
            temperature: contrôle la randomness (plus bas = plus déterministe)
            top_k: si fourni, ne garde que les k tokens les plus probables
        Returns:
            (batch, seq_len + max_new_tokens) séquence complète
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Tronquer si dépasse block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Prendre le dernier token
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
            
            # Top-k sampling (optionnel)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Échantillonnage
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Ajouter à la séquence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx