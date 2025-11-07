# Projet Transformer - Documentation Technique Complète

## Vue d'ensemble

Ce projet contient une implémentation pédagogique et modulaire de modèles Transformer pour différentes tâches : génération de texte (GPT-like), classification de sentiment et traduction automatique. Le code est organisé de manière modulaire pour faciliter la compréhension, l'entraînement, l'inférence et la maintenance.

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Préparer les données dans les dossiers `data/` :
   - `textgen_data.txt` : données pour la génération de texte
   - `sentiment_data.txt` : données pour la classification de sentiment (format: label\ttexte)
   - `translation_data.txt` : données pour la traduction (format: source\ttarget)

3. Configurer les hyperparamètres dans `config.yaml`

4. Lancer l'entraînement :
```bash
python train.py
```

5. Pour utiliser l'interface web Streamlit :
```bash
streamlit run ui_streamlit.py
```

---

## INDEXATION COMPLÈTE DU PROJET

### 1. FICHIERS ESSENTIELS

#### `train.py`
**Rôle principal** : Point d'entrée pour l'entraînement des trois modèles du projet.

**Fonctions** :
- `train_model(model, dataset, dataloader, optimizer, criterion, checkpoint_path, epochs, checkpoint_every, device)`
  - Gère la boucle d'entraînement principale
  - Sauvegarde les checkpoints périodiques et le meilleur modèle
  - Affiche la progression de l'entraînement

**Responsabilités** :
- Instancie et entraîne trois modèles séquentiellement :
  1. **Decoder (GPT)** : modèle de génération de texte sur `textgen_data.txt`
  2. **Encoder** : modèle de classification de sentiment sur `sentiment_data.txt`
  3. **TranslationModel** : modèle de traduction EN-FR sur `translation_data.txt`

---

#### `config.yaml`
**Rôle principal** : Fichier de configuration centralisé contenant tous les hyperparamètres du projet.

**Paramètres** :
- `data_path` : chemin vers les données d'entrée
- `batch_size` : taille des batches d'entraînement
- `block_size` : longueur maximale des séquences (contexte)
- `embed_dim` : dimension des embeddings de tokens
- `num_layers` : nombre de couches Transformer
- `num_heads` : nombre de têtes d'attention
- `ff_hidden_dim` : dimension cachée du réseau feedforward
- `dropout` : taux de dropout
- `lr` : taux d'apprentissage
- `epochs` : nombre d'époques d'entraînement
- `checkpoint_every` : fréquence de sauvegarde des checkpoints
- `weight_decay` : régularisation L2 pour l'optimiseur

---

#### `configurator.py`
**Rôle principal** : Charge et parse le fichier de configuration YAML.

**Classes** :
- `Config(path="config.yaml")`
  - Charge le fichier YAML et convertit son contenu en attributs d'objet
  - Permet d'accéder aux paramètres via `cfg.param_name`

---

### 2. MODÈLES

#### `model/transformer.py`
**Rôle principal** : Implémentation du modèle GPT (Decoder) pour la génération de texte.

**Classes** :
- `TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)`
  - Bloc Transformer standard avec architecture Pre-LN
  - Composition : MultiHeadSelfAttention + FeedForward avec résidus et normalisation
  - Méthode `forward(x)` : applique attention puis feedforward avec connexions résiduelles

- `GPT(vocab_size, block_size, embed_dim, num_layers, num_heads, ff_hidden_dim, dropout)`
  - Modèle GPT complet pour génération auto-régressive
  - Architecture : embeddings de tokens + encodage positionnel + N blocs Transformer + projection vers vocabulaire
  - Méthode `forward(idx, targets=None)` : propagation avant, retourne logits et loss si targets fournis
  - Méthode `generate(idx, max_new_tokens)` : génération séquentielle auto-régressive avec échantillonnage multinomial
  - Méthode `_init_weights(module)` : initialisation Xavier des poids du modèle

---

#### `model/encoder.py`
**Rôle principal** : Modèle Encoder pour classification de sentiment.

**Classes** :
- `Encoder(vocab_size, embed_dim, num_classes, num_layers, hidden_dim, dropout)`
  - Architecture basée sur convolutions 1D pour capturer les patterns locaux
  - Composition : embeddings + convolutions 1D + pooling global + couches linéaires
  - Méthode `forward(x, y=None)` : encode la séquence et prédit la classe de sentiment (négatif/neutre/positif)
  - Retourne logits et loss si labels fournis

---

#### `model/decoder.py`
**Rôle principal** : Placeholder pour un modèle Decoder spécifique (actuellement non utilisé).

**Classes** :
- `Decoder()`
  - Version debug avec une couche linéaire fictive
  - Note : le projet utilise `GPT` du module `transformer.py` pour la génération de texte

---

#### `model/translation.py`
**Rôle principal** : Modèle de traduction séquence-à-séquence.

**Classes** :
- `TranslationModel(vocab_src, vocab_tgt, embed_dim, block_size, hidden_dim, dropout)`
  - Architecture encodeur-décodeur simplifiée pour traduction
  - Encode la séquence source avec convolutions 1D + pooling
  - Génère les tokens cibles séquentiellement
  - Méthode `forward(src, tgt=None)` : encode source et prédit tokens cibles
  - Méthode `generate(src, max_length=None)` : génère la traduction complète token par token

---

#### `model/attention.py`
**Rôle principal** : Implémentation de l'attention multi-têtes avec masquage causal.

**Classes** :
- `MultiHeadSelfAttention(embed_dim, num_heads)`
  - Implémente l'attention scaled dot-product avec plusieurs têtes
  - Masquage causal pour préserver l'ordre temporel (GPT)
  - Méthode `forward(x)` : calcule Q, K, V, applique attention avec masque causal, combine les têtes

---

#### `model/feedforward.py`
**Rôle principal** : Réseau feedforward (FFN) utilisé dans chaque bloc Transformer.

**Classes** :
- `FeedForward(embed_dim, hidden_dim)`
  - Réseau à deux couches linéaires avec activation GELU
  - Expansion puis projection : embed_dim -> hidden_dim -> embed_dim
  - Méthode `forward(x)` : applique le réseau feedforward

---

#### `model/positional_encoding.py`
**Rôle principal** : Encodage positionnel sinusoïdal pour injecter l'information de position.

**Classes** :
- `PositionalEncoding(embed_dim, max_len)`
  - Génère les encodages positionnels sinusoïdaux (sin/cos)
  - Méthode `forward(x)` : ajoute les encodages positionnels aux embeddings de tokens

---

#### `model/normalization.py`
**Rôle principal** : Couche de normalisation personnalisée.

**Classes** :
- `LayerNorm(embed_dim, eps, bias)`
  - Wrapper autour de Layer Normalization avec contrôle optionnel du biais
  - Utilisée dans l'architecture Pre-LN des blocs Transformer
  - Méthode `forward(x)` : normalise sur la dernière dimension

---

### 3. TRAITEMENT DES DONNÉES

#### `tokenizer.py`
**Rôle principal** : Tokenizer au niveau caractère pour convertir texte en indices et vice-versa.

**Classes** :
- `CharTokenizer(text)`
  - Crée un vocabulaire basé sur les caractères uniques du texte
  - Attributs : `vocab_size`, `stoi` (char -> index), `itos` (index -> char)
  - Méthode `encode(s)` : convertit une chaîne de caractères en liste d'indices
  - Méthode `decode(t)` : convertit une liste d'indices en chaîne de caractères

---

#### `dataset.py`
**Rôle principal** : Dataset PyTorch pour génération de texte (language modeling).

**Classes** :
- `TextDataset(data, block_size)`
  - Dataset pour entraînement GPT avec prédiction du token suivant
  - Méthode `__len__()` : retourne le nombre d'échantillons possibles
  - Méthode `__getitem__(idx)` : retourne (x, y) où y est x décalé d'un token (prédiction next-token)

---

#### `sentiment_dataset.py`
**Rôle principal** : Dataset pour classification de sentiment.

**Classes** :
- `SentimentDataset(path, tokenizer, block_size)`
  - Charge les données depuis un fichier format "label\ttexte"
  - Labels : "negative" (0), "neutre" (1), "positive" (2)
  - Méthode `__getitem__(idx)` : retourne (x, y) où x est la séquence tokenisée et y est le label
  - Applique padding à droite si nécessaire

---

#### `translation_dataset.py`
**Rôle principal** : Dataset pour traduction séquence-à-séquence.

**Classes** :
- `TranslationDataset(path, tokenizer_src, tokenizer_tgt, block_size)`
  - Charge les paires source-cible depuis un fichier format "source\ttarget"
  - Utilise deux tokenizers distincts pour source et cible
  - Méthode `__getitem__(idx)` : retourne (src, tgt) tokenisées avec padding

---

#### `dataloader.py`
**Rôle principal** : Création de DataLoaders PyTorch standardisés.

**Fonctions** :
- `create_dataloader(dataset, batch_size)`
  - Crée un DataLoader PyTorch avec shuffle activé
  - Retourne un objet DataLoader pour l'itération en batches

---

### 4. ENTRAÎNEMENT ET OPTIMISATION

#### `engine.py`
**Rôle principal** : Boucle d'entraînement d'une époque.

**Fonctions** :
- `train_one_epoch(model, dataloader, optimizer, criterion, device)`
  - Itère sur un DataLoader pour une époque complète
  - Calcule la loss, fait la rétropropagation et met à jour les poids
  - Retourne la loss moyenne de l'époque
  - Affiche la progression avec tqdm

---

#### `optimizer/adamw.py`
**Rôle principal** : Création de l'optimiseur AdamW.

**Fonctions** :
- `create_optimizer(model, lr, weight_decay)`
  - Instancie un optimiseur AdamW avec les paramètres spécifiés
  - Configure betas=(0.9, 0.999) par défaut

---

#### `checkpoint_manager.py`
**Rôle principal** : Gestion de la sauvegarde et chargement des checkpoints.

**Fonctions** :
- `save_checkpoint(model, optimizer, epoch, path)`
  - Sauvegarde l'état du modèle, de l'optimiseur et l'époque dans un fichier .pt
  - Crée le dossier parent si nécessaire

- `load_checkpoint(model, optimizer, path)`
  - Charge un checkpoint depuis un fichier
  - Restaure les états du modèle et de l'optimiseur
  - Retourne le numéro d'époque sauvegardé

---

### 5. INFÉRENCE

#### `inference.py`
**Rôle principal** : Script d'inférence pour génération de texte avec le modèle GPT entraîné.

**Fonctions** :
- `generate(model, start_tokens, max_new_tokens, tokenizer, device)`
  - Génère du texte auto-régressivement à partir d'un prompt
  - Utilise échantillonnage multinomial après softmax
  - Retourne le texte décodé

**Processus** :
1. Charge la configuration et le tokenizer
2. Instancie le modèle GPT
3. Charge un checkpoint sauvegardé
4. Génère du texte à partir d'un prompt

---

#### `ui_streamlit.py`
**Rôle principal** : Interface web Streamlit pour tester les trois modèles.

**Fonctions** :
- `load_tokenizers()`
  - Charge et cache les tokenizers pour chaque tâche (décoder, encoder, traduction)

**Fonctionnalités** :
- **Génération de texte** : test du modèle GPT avec prompt personnalisable
- **Classification de sentiment** : analyse de texte avec probabilités par classe
- **Traduction** : traduction bidirectionnelle EN-FR avec affichage des résultats

---

### 6. UTILITAIRES

#### `common.py`
**Rôle principal** : Fonctions utilitaires générales.

**Fonctions** :
- `set_seed(seed)` : initialise les graines aléatoires pour reproductibilité (random, numpy, torch)
- `count_parameters(model)` : compte le nombre total de paramètres entraînables du modèle

---

#### `execution.py`
**Rôle principal** : Fonctions d'aide pour l'exécution depuis la ligne de commande.

**Fonctions** :
- `print_startup(cfg)` : affiche la configuration au démarrage de l'entraînement

---

#### `report.py`
**Rôle principal** : Fonctions de rapport et logging.

**Fonctions** :
- `simple_report(epoch, loss)` : affiche un rapport simple avec l'époque et la loss

---

#### `core_eval.py`
**Rôle principal** : Fonctions d'évaluation du modèle.

**Fonctions** :
- `ppl_from_loss(loss)` : calcule la perplexité à partir de la loss (exp(loss))

---

#### `loss_eval.py`
**Rôle principal** : Fonctions liées au calcul de loss.

**Fonctions** :
- `get_criterion()` : retourne une instance de CrossEntropyLoss pour l'entraînement

---

### 7. SCRIPTS AUXILIAIRES

#### `extract_vocab.py`
**Rôle principal** : Script utilitaire pour extraire les vocabulaires depuis un checkpoint.

**Fonctionnalités** :
- Analyse un checkpoint pour trouver les vocabulaires source/cible
- Sauvegarde les vocabulaires en fichiers pickle
- Affiche les informations sur les vocabulaires trouvés

---

## Structure du projet

```
transformer_project/
├── train.py                 # Entraînement des 3 modèles
├── inference.py             # Génération de texte
├── config.yaml              # Configuration centralisée
├── configurator.py          # Chargeur de configuration
├── tokenizer.py             # Tokenizer caractère
├── dataset.py               # Dataset génération de texte
├── sentiment_dataset.py     # Dataset classification
├── translation_dataset.py   # Dataset traduction
├── dataloader.py            # Création DataLoaders
├── engine.py                # Boucle d'entraînement
├── checkpoint_manager.py    # Gestion checkpoints
├── common.py                # Utilitaires généraux
├── execution.py             # Helpers exécution
├── report.py                # Logging et rapports
├── core_eval.py             # Métriques d'évaluation
├── loss_eval.py             # Fonctions de loss
├── extract_vocab.py         # Extraction vocabulaires
├── ui_streamlit.py          # Interface web
├── model/                   # Module modèles
│   ├── transformer.py       # GPT (Decoder)
│   ├── encoder.py           # Encoder classification
│   ├── decoder.py           # Decoder (placeholder)
│   ├── translation.py       # Modèle traduction
│   ├── attention.py         # Multi-head attention
│   ├── feedforward.py       # Réseau feedforward
│   ├── positional_encoding.py # Encodage positionnel
│   └── normalization.py     # Layer normalization
├── optimizer/               # Module optimiseurs
│   └── adamw.py             # Optimiseur AdamW
├── data/                    # Données d'entraînement
│   ├── textgen_data.txt
│   ├── sentiment_data.txt
│   └── translation_data.txt
└── checkpoints/             # Checkpoints sauvegardés
```

---

## Utilisation

### Entraînement

Lancer l'entraînement de tous les modèles :
```bash
python train.py
```

Cela entraînera séquentiellement :
1. Le modèle Decoder (GPT) pour génération de texte
2. Le modèle Encoder pour classification de sentiment
3. Le modèle TranslationModel pour traduction EN-FR

### Inférence

Générer du texte avec le modèle GPT :
```bash
python inference.py
```

### Interface web

Lancer l'interface Streamlit pour tester interactivement les modèles :
```bash
streamlit run ui_streamlit.py
```

---

## Architecture des modèles

### GPT (Decoder)
- Architecture : Transformer decoder-only avec attention causale
- Usage : Génération de texte auto-régressive
- Données : `data/textgen_data.txt`

### Encoder
- Architecture : Convolutions 1D + pooling global + couches linéaires
- Usage : Classification de sentiment (3 classes)
- Données : `data/sentiment_data.txt`

### TranslationModel
- Architecture : Encodeur-décodeur simplifié avec convolutions
- Usage : Traduction séquence-à-séquence EN-FR
- Données : `data/translation_data.txt`

---

## Configuration

Tous les hyperparamètres sont centralisés dans `config.yaml`. Modifier ce fichier pour ajuster :
- Architecture du modèle (dimensions, nombre de couches)
- Paramètres d'entraînement (batch size, learning rate, epochs)
- Chemins des données et checkpoints

---

## Dépendances

Voir `requirements.txt` pour la liste complète. Principales dépendances :
- `torch` : Framework deep learning
- `tqdm` : Barres de progression
- `pyyaml` : Parsing de configuration
- `streamlit` : Interface web

---

## Notes de développement

- Le projet utilise des tokenizers au niveau caractère pour simplifier
- Les modèles Encoder et TranslationModel utilisent des architectures simplifiées (convolutions) plutôt que des Transformers complets
- Le modèle Decoder utilise une architecture Pre-LN pour plus de stabilité
- Les checkpoints sont sauvegardés périodiquement et lors de meilleures performances

---

## Auteur

Réalisé par Ourti Abdelilah

---

*Documentation générée pour faciliter la compréhension et la maintenance du projet Transformer.*
