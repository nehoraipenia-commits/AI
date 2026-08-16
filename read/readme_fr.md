# AI - Neural Generative Core & Semantic Framework

Un système avancé, léger et hybride d'intelligence générative, développé nativement avec PyTorch. Le framework combine une architecture Transformer causale personnalisée avec le Byte-Pair Encoding (BPE), la recherche vectorielle sémantique (RAG), la reconstruction de phrases françaises (FSR), le suivi dynamique du profil utilisateur et le fine-tuning coopératif hors ligne de LLM locaux via des instances Ollama.

### Fonctionnalités principales

* Moteur Transformer causal personnalisé : Entièrement développé à partir des primitives PyTorch, avec encodages positionnels sinusoïdaux, attention causale multi-têtes et normalisation des couches.

* Tokenisation Byte-Pair Encoding (BPE) : Pipeline personnalisé de tokenisation par sous-mots avec mise à l'échelle dynamique du vocabulaire et normalisation complète de l'Unicode et des accents.

* Mémoire sémantique multi-fichiers (RAG) : Magasin vectoriel basé sur TF-IDF et la similarité cosinus, capable d'indexer des bases de code Python (.py) et de la documentation (.txt) pour une génération prenant en compte le contexte.

* Intégration Ollama & fine-tuning dynamique : Framework multi-modèles coopératif qui récupère des extensions de connaissances auprès de LLM locaux (par exemple Gemma, Llama) afin d'entraîner dynamiquement les poids PyTorch personnalisés hors ligne.

* Reconstruction de phrases françaises (FSR) : Module neuronal Seq2Seq dédié à la correction des contractions françaises, des erreurs syntaxiques et des fautes de frappe.

* Mémoire utilisateur dynamique & moteur de profil : Extraction en temps réel des informations utilisateur (nom, préférences, contexte), maintenues à travers des sessions de discussion successives.

* Interface CLI interactive : Interface terminal complète avec analyses d'exécution en temps réel et gestion de l'état du modèle.

* Architecture du système

```text
                                  +---------------------------+
                                  |    Entrée utilisateur /   |
                                  |          Prompt           |
                                  +-------------+-------------+
                                                |
                                                v
   +----------------------------+   +-----------+-----------+   +----------------------------+
   | Mémoire utilisateur        |-->| Mémoire sémantique    |<--| Fichiers du répertoire    |
   | dynamique                  |   | (RAG)                 |   | indexés (.py & .txt)      |
   | (faits & historique)       |   | (Magasin TF-IDF)      |   |                            |
   +----------------------------+   +-----------+-----------+   +----------------------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Tokenisation Byte-Pair     |
                                 | Encoding (BPE)              |
                                 | Pipeline                    |
                                 +--------------+--------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Transformer causal         |
                                 | personnalisé               |
                                 | (Attention multi-têtes)    |
                                 +--------------+--------------+
                                                |
                                                v
  +----------------------------+    +-----------+-------------+   +----------------------------+
  | Module de reconstruction   |<-- | Moteur de sampling      |--> | LLM Ollama local          |
  | française                  |    | intelligent                  | (Boucles de warmup        |
  | (Correction grammaticale)  |    | (Top-K/Nucleus Sampling) |   | coopératives)             |
  +----------------------------+    +-----------+-------------+   +----------------------------+
                                                |
                                                v
                                  +-------------+-------------+
                                  |  Réponse générée par      |
                                  |           l'IA             |
                                  +---------------------------+
```

## Prérequis

* Avant l'installation, assurez-vous que les logiciels suivants sont installés sur votre machine :

* Python : Version 3.1 ou supérieure

* PyTorch : Version 1.12.0 ou supérieure (support CUDA/MPS optionnel mais recommandé)

* Ollama (Optionnel) : Nécessaire uniquement si vous souhaitez utiliser le fine-tuning coopératif avec des LLM locaux.

* Installation & configuration

### Cloner le dépôt

```bash
git clone https://github.com/nehoraipenia-commits/AI.git
cd AI
```

### Créer et activer un environnement virtuel (Recommandé)

## Sur macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Sur Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Installer les dépendances requises

```bash
pip install torch numpy scikit-learn requests
```

ou

```bash
pip install -r requirements.txt
```

### Lancer l'interface CLI de l'assistant

```bash
python3 ai_pytorch.py
```

## Commandes de l'interface CLI interactive

Une fois le CLI lancé (`AI-CLI >>>`), vous pouvez entrer des prompts textuels standards ou utiliser les commandes de contrôle du système :

Commande

Description

```text
/status
```

Affiche les caractéristiques matérielles, l'allocation du périphérique, la taille du vocabulaire BPE actif et l'état du fichier de base mémoire.

```text
/train <epochs>
```

Réoptimise les poids du Transformer directement à partir de votre base de code et de votre documentation locales.

```text
/model <name>
```

Change l'identifiant du modèle Ollama local actif (par exemple, `/model gemma2:2b` ou `/model llama3`).

```text
/gemma <count>
```

Lance des boucles de génération coopérative avec Ollama afin d'étendre la base de connaissances locale et de réentraîner le modèle.

```text
/train_fsr <ep>
```

Entraîne le module Transformer de reconstruction de phrases françaises sur les textes locaux.

```text
/reconstruct <text>
```

Corrige les formulations françaises corrompues, les fautes de frappe ou les contractions à l'aide du modèle FSR.

```text
/save
```

Exporte manuellement les checkpoints et les poids actuels du système sur le disque (`checkpoints/`).

```text
/load
```

Recharge les états sauvegardés du système et les vocabulaires du tokenizer à partir des fichiers de checkpoint.

```text
/clear
```

Efface l'écran de l'interface terminal.

```text
/exit
```

Arrête proprement les processus et sauvegarde les poids actuels.

## Structure du dépôt

```text
AI/
├── AI.py                            # Contrôleur principal du pipeline système & CLI interactive
├── AdvancedPhraseReformulator.py    # Moteur de reformulation grammaticale basé sur les POS
├── ai_generation_helper.py          # Échantillonnage intelligent des logits, moteur de Markov & optimisation du texte
├── FrenchContractionHandler.py      # Règles de contraction française & reconstruction Seq2Seq
├── dynamic_user_memory.py           # Extracteur dynamique de faits utilisateur & gestion de l'historique
├── nlp_tokenization_suite.py        # Suite de tokenisation Byte-Pair Encoding (BPE)
├── big_book.txt                     # Corpus principal du système & base de connaissances
├── ollama_responses.txt             # Journal des extensions générées par les LLM coopératifs
└── checkpoints/                     # Répertoire de sérialisation des poids du modèle
```

Pour plus d'informations : https://nexeo-ai.netlify.app/
