# AI - Neural Generative Core & Semantic Framework

An advanced, lightweight hybrid generative intelligence system built natively with PyTorch. The framework combines a custom Causal Transformer architecture with Byte-Pair Encoding (BPE), semantic vector search (RAG), French Sentence Reconstruction (FSR), dynamic user profile tracking, and cooperative offline LLM fine-tuning via local Ollama instances.

### Key Features

- Custom Causal Transformer Engine: Built entirely from scratch using PyTorch primitives, featuring Sinusoidal Positional Encodings, Causal Multi-Head Self-Attention, and Layer Normalization.

- Byte-Pair Encoding (BPE) Tokenization: Custom subword tokenization pipeline with dynamic vocabulary scaling and full Unicode/accent normalization.

- Multi-File Semantic Memory (RAG): TF-IDF & Cosine Similarity vector store capable of indexing Python codebases (.py) and documentation (.txt) for context-aware generation.

- Ollama Integration & Dynamic Fine-Tuning: Cooperative multi-model framework that pulls knowledge expansions from local LLMs (e.g., Gemma, Llama) to dynamically train the custom PyTorch weights offline.

- French Sentence Reconstruction (FSR): Dedicated Seq2Seq neural module for denoising, fixing French contractions, syntax errors, and typo handling.

- Dynamic User Memory & Profile Engine: Real-time extraction of user facts (name, preferences, context) maintained across rolling chat sessions.

- Interactive CLI Interface: Feature-rich terminal interface with live execution analytics and model state management.

- System Architecture

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
 +----------------------------+   +-----------+-----------+   +----------------------------+
 | French Reconstruction Module|<--| Smart Sampler Engine  |-->| Ollama Local LLM           |
 | (FSR Grammar Correction)   |   | (Top-K / Nucleus Sampling)|   | (Cooperative Warmup Loops) |
 +----------------------------+   +-----------+-----------+   +----------------------------+
                                              |
                                              v
                                +-------------+-------------+
                                |  Generated AI Response    |
                                +---------------------------+

```

## Prerequisites

- Before installing, make sure you have the following software installed on your machine:

- Python: Version 3.8 or higher

- PyTorch: Version 1.12.0 or higher (CUDA/MPS support optional but recommended)

- Ollama (Optional): Required only if you intend to run cooperative LLM fine-tuning locally.

- Installation & Setup

### Clone the repository

```
git clone https://github.com/nehoraipenia-commits/AI.git
cd AI
```


### Create and activate a virtual environment (Recommended)

## On macOS/Linux
python3 -m venv venv
source venv/bin/activate

## On Windows

```
python -m venv venv
venv\Scripts\activate
```

### Install required dependencies

```
pip install torch numpy scikit-learn requests
```

### Launch the Assistant CLI

```
python3 ai_pytorch.py
```

Interactive CLI Commands

Once the CLI is running (AI-CLI >>>), you can enter standard text prompts or utilize system control commands:

Command

Description

```
/status
```


View hardware specs, device allocation, active BPE vocabulary size, and memory base file status.

```
/train <epochs>
```


Re-optimize Transformer weights directly on your local codebase and documentation.

```
/model <name>
```


Change the active local Ollama model identifier (e.g., /model gemma2:2b or /model llama3).

```
/gemma <count>
```


Run cooperative generation loops with local Ollama to expand local knowledge base and retrain model.

```
/train_fsr <ep>
```


Train the French Sentence Reconstruction Transformer module on local texts.

```
/reconstruct <text>
```


Correct corrupted French phrasing, typos, or contractions using the FSR model.

```
/save
```


Manually export current brain checkpoints and weights to disk (checkpoints/).

```
/load
```


Reload saved system states and tokenizer vocabularies from checkpoint files.

```
/clear
```


Clear the terminal interface screen.

```
/exit
```


Safely terminate processes and save current weights.

Repository Structure

```
AI/
├── AI.py                            # Main system pipeline controller & interactive CLI
├── AdvancedPhraseReformulator.py    # POS-aware grammatical reformulation engine
├── ai_generation_helper.py          # Smart logits sampler, Markov engine & text optimizer
├── FrenchContractionHandler.py      # French contraction rules & Seq2Seq reconstruction
├── dynamic_user_memory.py           # Dynamic user fact extractor & rolling history manager
├── nlp_tokenization_suite.py        # Subword Byte-Pair Encoding (BPE) tokenizer suite
├── big_book.txt                     # Primary system corpus & knowledge store
├── ollama_responses.txt             # Cooperative LLM expansion logging store
└── checkpoints/                     # Model weight serialization directory
```

