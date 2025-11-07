import streamlit as st
import torch
from configurator import Config
from tokenizer import CharTokenizer
from model.transformer import GPT
from model.encoder import Encoder
from model.translation import TranslationModel
import os

st.set_page_config(page_title="Transformer Project - Interface de Test", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialisation des tokenizers
@st.cache_resource
def load_tokenizers():
    """Charge les tokenizers pour chaque tâche"""
    tokenizers = {}
    
    # Tokenizer pour génération (Decoder)
    try:
        with open("data/textgen_data.txt", "r", encoding="utf-8") as f:
            text_gen = f.read()
        tokenizers['decoder'] = CharTokenizer(text_gen)
    except:
        tokenizers['decoder'] = None
    
    # Tokenizer pour sentiment (Encoder)
    try:
        with open("data/sentiment_data.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            text_sent = " ".join([l.split("\t", 1)[1] if "\t" in l else l.strip() for l in lines])
        tokenizers['encoder'] = CharTokenizer(text_sent)
    except:
        tokenizers['encoder'] = None
    
    # Tokenizers pour traduction (source et target)
    try:
        with open("data/translation_data.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            text_src = " ".join([l.split("\t", 1)[0] for l in lines if "\t" in l])
            text_tgt = " ".join([l.split("\t", 1)[1] for l in lines if "\t" in l])
        tokenizers['translation_src'] = CharTokenizer(text_src)
        tokenizers['translation_tgt'] = CharTokenizer(text_tgt)
    except:
        tokenizers['translation_src'] = None
        tokenizers['translation_tgt'] = None
    
    return tokenizers

tokenizers = load_tokenizers()

# Titre principal
st.title("🚀 Interface de Test des Modèles Transformer")
st.markdown("---")

# Sidebar pour sélection de la tâche
task = st.sidebar.selectbox(
    "Sélectionner la tâche",
    ["Génération de texte (Decoder)", "Classification de sentiment (Encoder)", "Traduction EN-FR (TranslationModel)"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Device:** {device}")

# ============================================
# 1. GÉNÉRATION DE TEXTE (Decoder)
# ============================================
if task == "Génération de texte (Decoder)":
    st.header("📝 Génération de texte")
    
    checkpoint_path = st.text_input("Chemin du checkpoint", value="checkpoints/decoder_ckpt.pt")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_area("Prompt initial", value="Once upon a time", height=100)
        num_tokens = st.number_input("Nombre de tokens à générer", min_value=10, max_value=500, value=100)
    
    with col2:
        cfg_path = st.text_input("Chemin config.yaml", value="config.yaml")
    
    if st.button("🔮 Générer du texte"):
        if not os.path.exists(checkpoint_path):
            st.error(f"❌ Checkpoint introuvable : {checkpoint_path}")
        elif tokenizers['decoder'] is None:
            st.error("❌ Tokenizer pour génération non disponible")
        else:
            try:
                with st.spinner("Chargement du modèle..."):
                    cfg = Config(cfg_path)
                    model = GPT(
                        vocab_size=tokenizers['decoder'].vocab_size,
                        block_size=cfg.block_size,
                        embed_dim=cfg.embed_dim,
                        num_layers=cfg.num_layers,
                        num_heads=cfg.num_heads,
                        ff_hidden_dim=cfg.ff_hidden_dim,
                        dropout=cfg.dropout
                    ).to(device)
                    
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model"])
                    model.eval()
                
                with st.spinner("Génération en cours..."):
                    start_tokens = tokenizers['decoder'].encode(prompt)
                    idx = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)
                    
                    for _ in range(num_tokens):
                        idx_cond = idx[:, -cfg.block_size:]
                        logits, _ = model(idx_cond, targets=None)
                        logits = logits[:, -1, :]
                        probs = torch.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)
                    
                    generated = tokenizers['decoder'].decode(idx[0].tolist())
                
                st.success("✅ Génération terminée !")
                
                # Affichage amélioré
                st.markdown("### 📝 Prompt initial :")
                st.info(prompt)
                
                st.markdown("### ✨ Texte généré :")
                st.code(generated, language=None)
                
                # Informations supplémentaires
                with st.expander("ℹ️ Informations sur la génération"):
                    st.write(f"**Nombre de tokens générés :** {num_tokens}")
                    st.write(f"**Longueur totale du texte :** {len(generated)} caractères")
                    st.write(f"**Vocabulaire utilisé :** {tokenizers['decoder'].vocab_size} caractères uniques")
                    st.write(f"**Block size :** {cfg.block_size}")
                
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# ============================================
# 2. CLASSIFICATION DE SENTIMENT (Encoder)
# ============================================
elif task == "Classification de sentiment (Encoder)":
    st.header("😊 Classification de sentiment")
    
    checkpoint_path = st.text_input("Chemin du checkpoint", value="checkpoints/encoder_ckpt.pt")
    
    text_input = st.text_area("Texte à analyser", value="I am very disappointed with the result", height=100)
    
    # Charger la config pour obtenir le block_size par défaut
    cfg = Config()
    block_size = st.number_input("Block size", min_value=16, max_value=256, value=cfg.block_size)
    
    if st.button("🔍 Classifier"):
        if not os.path.exists(checkpoint_path):
            st.error(f"❌ Checkpoint introuvable : {checkpoint_path}")
        elif tokenizers['encoder'] is None:
            st.error("❌ Tokenizer pour sentiment non disponible")
        else:
            try:
                with st.spinner("Chargement du modèle..."):
                    cfg = Config()
                    model = Encoder(
                        vocab_size=tokenizers['encoder'].vocab_size,
                        embed_dim=cfg.embed_dim,
                        num_classes=3,
                        num_layers=2,
                        hidden_dim=cfg.ff_hidden_dim,
                        dropout=cfg.dropout
                    ).to(device)
                    
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model"])
                    model.eval()
                
                with st.spinner("Analyse en cours..."):
                    ids = tokenizers['encoder'].encode(text_input)
                    ids = ids[:block_size]
                    if len(ids) < block_size:
                        ids = ids + [0] * (block_size - len(ids))
                    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                    
                    logits, _ = model(x, y=None)
                    probs = torch.softmax(logits, dim=-1)
                    pred_class = torch.argmax(logits, dim=-1).item()
                
                labels = ["Négatif", "Neutre", "Positif"]
                idx2label = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                
                st.success(f"✅ Classification terminée !")
                
                # Afficher les résultats avec une meilleure présentation
                col1, col2, col3 = st.columns(3)
                with col1:
                    prob_neg = probs[0][0].item()
                    st.metric("😟 Négatif", f"{prob_neg:.2%}", 
                             delta=f"{prob_neg*100:.1f}%")
                with col2:
                    prob_neu = probs[0][1].item()
                    st.metric("😐 Neutre", f"{prob_neu:.2%}", 
                             delta=f"{prob_neu*100:.1f}%")
                with col3:
                    prob_pos = probs[0][2].item()
                    st.metric("😊 Positif", f"{prob_pos:.2%}", 
                             delta=f"{prob_pos*100:.1f}%")
                
                # Afficher la prédiction principale
                st.markdown(f"### 🎯 Résultat : **{idx2label[pred_class]}**")
                
                # Barre de progression pour chaque classe
                st.markdown("#### 📊 Probabilités détaillées :")
                st.progress(prob_neg, text="Négatif")
                st.progress(prob_neu, text="Neutre")
                st.progress(prob_pos, text="Positif")
                
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# ============================================
# 3. TRADUCTION EN-FR (TranslationModel)
# ============================================
elif task == "Traduction EN-FR (TranslationModel)":
    st.header("🌐 Traduction Anglais ↔ Français")
    
    checkpoint_path = st.text_input("Chemin du checkpoint", value="checkpoints/translation_ckpt.pt")
    
    col1, col2 = st.columns(2)
    
    with col1:
        direction = st.radio("Direction", ["Anglais → Français", "Français → Anglais"])
        text_input = st.text_area("Texte à traduire", value="Hello, how are you?", height=100)
    
    with col2:
        # Charger la config pour obtenir le block_size par défaut
        cfg_default = Config()
        block_size = st.number_input("Block size", min_value=16, max_value=256, value=cfg_default.block_size)
    
    if st.button("🔄 Traduire"):
        if not os.path.exists(checkpoint_path):
            st.error(f"❌ Checkpoint introuvable : {checkpoint_path}")
        elif tokenizers['translation_src'] is None or tokenizers['translation_tgt'] is None:
            st.error("❌ Tokenizers pour traduction non disponibles")
        else:
            try:
                with st.spinner("Chargement du modèle..."):
                    cfg = Config()
                    tokenizer_src = tokenizers['translation_src'] if direction == "Anglais → Français" else tokenizers['translation_tgt']
                    tokenizer_tgt = tokenizers['translation_tgt'] if direction == "Anglais → Français" else tokenizers['translation_src']
                    
                    model = TranslationModel(
                        vocab_src=tokenizer_src.vocab_size,
                        vocab_tgt=tokenizer_tgt.vocab_size,
                        embed_dim=cfg.embed_dim,
                        block_size=block_size,
                        hidden_dim=cfg.ff_hidden_dim,
                        dropout=cfg.dropout
                    ).to(device)
                    
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model"])
                    model.eval()
                
                with st.spinner("Traduction en cours..."):
                    ids = tokenizer_src.encode(text_input)
                    # Utiliser block_size du modèle ou celui de l'interface
                    max_block = min(block_size, cfg.block_size)
                    ids = ids[:max_block]
                    if len(ids) < max_block:
                        ids = ids + [0] * (max_block - len(ids))
                    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                    
                    # Générer la séquence de traduction
                    generated_tokens = model.generate(x, max_length=max_block)
                    
                    # Décoder les tokens générés
                    # Filtrer les tokens invalides (0 et tokens hors vocabulaire)
                    valid_tokens = [t for t in generated_tokens if t != 0 and t < tokenizer_tgt.vocab_size]
                    if valid_tokens:
                        translated_text = tokenizer_tgt.decode(valid_tokens)
                        # Nettoyer le texte (enlever caractères spéciaux étranges)
                        translated_text = translated_text.strip()
                    else:
                        translated_text = "[Aucune traduction générée - le modèle doit être réentraîné]"
                
                st.success("✅ Traduction terminée !")
                
                # Affichage amélioré avec deux colonnes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📝 Texte original")
                    st.info(f"**{direction.split('→')[0].strip()}**")
                    st.code(text_input, language=None)
                
                with col2:
                    st.markdown("### 🌐 Traduction")
                    st.info(f"**{direction.split('→')[1].strip()}**")
                    if translated_text and not translated_text.startswith("["):
                        st.success(translated_text)
                        st.code(translated_text, language=None)
                    else:
                        st.warning(translated_text)
                
                # Informations de debug (collapsible)
                with st.expander("🔍 Informations techniques"):
                    st.write(f"**Tokens générés :** {len(generated_tokens)}")
                    st.write(f"**Tokens valides :** {len(valid_tokens)}")
                    st.write(f"**Vocabulaire source :** {tokenizer_src.vocab_size}")
                    st.write(f"**Vocabulaire cible :** {tokenizer_tgt.vocab_size}")
                    if valid_tokens:
                        st.write(f"**Tokens (premiers 20) :** {valid_tokens[:20]}")
                    
                if not valid_tokens or translated_text.startswith("["):
                    st.warning("⚠️ **Note :** Le modèle n'a pas généré de traduction valide. Cela peut indiquer que :")
                    st.write("- Le modèle n'a pas été suffisamment entraîné")
                    st.write("- Le vocabulaire source/cible ne correspond pas aux données d'entraînement")
                    st.write("- Les données d'entraînement sont insuffisantes")
                    st.write("**Solution :** Relancer l'entraînement avec `python train.py`")
                
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Transformer Project** - Interface Streamlit pour tester les modèles")

