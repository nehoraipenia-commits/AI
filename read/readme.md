# IA - Noyau génératif neuronal et cadre sémantique

Un système d'intelligence générative hybride avancé et léger, conçu nativement avec PyTorch. Ce cadre combine une architecture de transformateur causal personnalisée avec l'encodage par paires d'octets (BPE), la recherche vectorielle sémantique (RAG), la reconstruction de phrases françaises (FSR), le suivi dynamique des profils utilisateurs et l'ajustement coopératif hors ligne du modèle linéaire de base (LLM) via des instances locales d'Ollama.

## Fonctionnalités clés

- Moteur de transformateur causal personnalisé : Entièrement conçu à partir de zéro avec les primitives PyTorch, il intègre l'encodage positionnel sinusoïdal, l'auto-attention causale multi-têtes et la normalisation des couches.

- Tokenisation par encodage par paires d'octets (BPE) : Pipeline de tokenisation de sous-mots personnalisé avec mise à l'échelle dynamique du vocabulaire et normalisation complète Unicode/accent.

- Mémoire sémantique multi-fichiers (RAG) : Stockage vectoriel de similarité TF-IDF et cosinus capable d'indexer les bases de code Python (.py) et la documentation (.txt) pour une génération contextuelle.

- Intégration d'Ollama et ajustement dynamique : Cadre multi-modèles coopératif qui exploite les extensions de connaissances des LLM locaux (par exemple, Gemma, Llama) pour entraîner dynamiquement hors ligne les poids PyTorch personnalisés.

- Reconstruction de phrases françaises (FSR) : Module neuronal Seq2Seq dédié au débruitage, à la correction des contractions françaises, des erreurs de syntaxe et des fautes de frappe.

- Moteur de mémoire utilisateur dynamique et de profils : Extraction en temps réel des informations utilisateur (nom, préférences, contexte) conservées au fil des sessions de chat.

- Interface CLI interactive : Interface terminal riche en fonctionnalités avec analyses d'exécution en direct et gestion de l'état du modèle.

- Architecture système

```
                                  +---------------------------+
                                  |    User Input / Prompt    |
                                  +-------------+-------------+
                                                |
                                                v
   +----------------------------+   +-----------+-----------+   +----------------------------+
   | Dynamic User Memory Core   |-->| Semantic Memory (RAG) |<--| Indexed Directory Files    |
   | (Facts & History Context)  |   | (TF-IDF Vector Store) |   | (.py code & .txt docs)     |
   +----------------------------+   +-----------+-----------+   +----------------------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Byte-Pair Encoding (BPE)    |
                                 | Tokenizer Pipeline          |
                                 +--------------+--------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Custom Causal Transformer   |
                                 | (Multi-Head Self-Attention) |
                                 +--------------+--------------+
                                                |
                                                v
  +----------------------------+    +-----------+-------------+   +----------------------------+
  | French Reconstruction Module|<--| Smart Sampler Engine    |-->| Ollama Local LLM           |
  | (FSR Grammar Correction)    |   | (Top-K/Nucleus Sampling)|   | (Cooperative Warmup Loops) |
  +----------------------------+    +-----------+-------------+   +----------------------------+
                                                |
                                                v
                                  +-------------+-------------+
                                  |  Generated AI Response    |
                                  +---------------------------+


```

## Prérequis

- Avant l'installation, assurez-vous d'avoir installé les logiciels suivants sur votre machine :

- Python : Version 3.8 ou supérieure

- PyTorch : Version 1.12.0 ou supérieure (prise en charge CUDA/MPS optionnelle mais recommandée)

- Ollama (optionnel) : Requis uniquement si vous souhaitez effectuer un réglage fin LLM coopératif en local.

Installation et configuration

### Cloner le dépôt

```
git clone https://github.com/nehoraipenia-commits/AI.git
cd AI
```

### Créer et activer un environnement virtuel (recommandé)

## Sous macOS/Linux

python3 -m venv venv
source venv/bin/activate

## Sous Windows

```
python -m venv venv
venv\Scripts\activate
```

### Installer les dépendances requises

```
pip install torch numpy scikit-learn requests
```

### Lancer l'interface de ligne de commande de l'assistant

```
python3 ai_pytorch.py
```

## Commandes interactives de l'interface de ligne de commande

Une fois l'interface de ligne de commande lancée (AI-CLI >>>), vous pouvez saisir des invites de texte standard ou utiliser les commandes de contrôle système :

Commande

Description

```
/status
```

Afficher le matériel Spécifications, allocation des ressources, taille du vocabulaire BPE actif et état des fichiers de base en mémoire.

```
/train <epochs>
```

Réoptimisez les poids du Transformer directement dans votre code source local et votre documentation.

```
/model <name>
```

Changez le modèle Ollama local actif.
