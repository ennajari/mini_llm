# model/translation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward
from .positional_encoding import PositionalEncoding
from .normalization import LayerNorm


class TransformerEncoderBlock(nn.Module):
    """Bloc encoder pour le modèle seq2seq"""
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class CrossAttention(nn.Module):
    """Cross-attention pour permettre au decoder d'attendre l'encoder"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, encoder_output):
        """
        Args:
            x: (B, T_tgt, C) - decoder queries
            encoder_output: (B, T_src, C) - encoder keys/values
        """
        B, T_tgt, C = x.shape
        T_src = encoder_output.size(1)
        
        # Queries du decoder
        q = self.q_proj(x).reshape(B, T_tgt, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, nh, T_tgt, hd)
        
        # Keys et Values de l'encoder
        kv = self.kv_proj(encoder_output).reshape(B, T_src, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.permute(0, 2, 1, 3)  # (B, nh, T_src, hd)
        v = v.permute(0, 2, 1, 3)  # (B, nh, T_src, hd)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = F.softmax(att, dim=-1)
        
        out = att @ v  # (B, nh, T_tgt, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, T_tgt, C)
        
        return self.out_proj(out)


class TransformerDecoderBlock(nn.Module):
    """Bloc decoder avec self-attention ET cross-attention"""
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.ln3 = LayerNorm(embed_dim)
        
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        # Self-attention (causal)
        x = x + self.dropout(self.self_attn(self.ln1(x)))
        # Cross-attention avec encoder
        x = x + self.dropout(self.cross_attn(self.ln2(x), encoder_output))
        # Feed-forward
        x = x + self.dropout(self.ff(self.ln3(x)))
        return x


class TranslationModel(nn.Module):
    """
    Modèle Seq2Seq Transformer complet pour traduction.
    Architecture encoder-decoder avec attention croisée.
    """
    def __init__(self, vocab_src, vocab_tgt, embed_dim=256, block_size=512, 
                 num_layers=6, num_heads=8, ff_hidden_dim=1024, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        self.block_size = block_size
        self.vocab_tgt = vocab_tgt
        
        # Embeddings source et target
        self.src_embedding = nn.Embedding(vocab_src, embed_dim)
        self.tgt_embedding = nn.Embedding(vocab_tgt, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, block_size)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.encoder_ln = LayerNorm(embed_dim)
        self.decoder_ln = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Tête de sortie
        self.lm_head = nn.Linear(embed_dim, vocab_tgt, bias=False)
        
        # Initialisation
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src):
        """Encode la séquence source"""
        x = self.src_embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        return self.encoder_ln(x)
    
    def decode(self, tgt, encoder_output):
        """Decode avec attention vers l'encoder"""
        x = self.tgt_embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.decoder_blocks:
            x = block(x, encoder_output)
        
        return self.decoder_ln(x)
    
    def forward(self, src, tgt=None):
        """
        Args:
            src: (batch, src_len) séquence source
            tgt: (batch, tgt_len) séquence cible (optionnel)
        Returns:
            logits: (batch, tgt_len, vocab_tgt)
            loss: scalaire si tgt fourni
        """
        # Encoder
        encoder_output = self.encode(src)
        
        if tgt is None:
            # Mode génération (voir méthode generate)
            return encoder_output, None
        
        # Decoder
        decoder_output = self.decode(tgt, encoder_output)
        logits = self.lm_head(decoder_output)
        
        # Calculer la perte
        # On décale les targets d'une position (predict next token)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_tgt),
            tgt[:, 1:].reshape(-1)
        )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, src, max_length=50, temperature=1.0, eos_token=0):
        """
        Génération autoregressive de traduction.
        
        Args:
            src: (batch, src_len) séquence source
            max_length: longueur max de la traduction
            temperature: contrôle la diversité
            eos_token: token de fin de séquence
        Returns:
            (batch, gen_len) traduction générée
        """
        self.eval()
        batch_size = src.size(0)
        
        # Encoder une seule fois
        encoder_output = self.encode(src)
        
        # Commencer avec token BOS (supposons que c'est 1)
        generated = torch.ones((batch_size, 1), dtype=torch.long, device=src.device)
        
        for _ in range(max_length):
            # Decoder
            decoder_output = self.decode(generated, encoder_output)
            logits = self.lm_head(decoder_output[:, -1, :]) / temperature
            
            # Échantillonnage
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Arrêter si tous les batchs ont généré EOS
            if (next_token == eos_token).all():
                break
        
        return generated