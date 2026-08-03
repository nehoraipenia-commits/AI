import os
import sys
import math
import json
import time
import random
import requests
import glob
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Iterator

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Scikit-learn Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the premium NLP tokenization suite provided in the workspace
try:
    from nlp_tokenization_suite import (
        TokenizerConfig,
        BytePairEncodingTokenizer,
        TokenizedDataset,
        DataCollatorWithPadding,
        TokenEmbedding
    )
except ImportError:
    # Fail-safe structural definitions if imports are disrupted
    print("[Error] nlp_tokenization_suite.py not found or failed to load. Please make sure it is in the same directory.")
    sys.exit(1)

# Import the French Sentence Reconstruction Library (FrenchContractionHandler.py)
try:
    from FrenchContractionHandler import (
        FrenchContractionHandler,
        FrenchSentenceTokenizer,
        Vocabulary as FSR_Vocabulary,
        FrenchSentenceReconstructor,
        SentenceCorruptor
    )
    FSR_AVAILABLE = True
except ImportError:
    # Set to False if FrenchContractionHandler is missing or has a different path
    FSR_AVAILABLE = False

# Import our new custom AI generation helper module
try:
    from ai_generation_helper import SmartSampler, FailsafeMarkovEngine, AIOutputOptimizer
    HELPER_AVAILABLE = True
except ImportError:
    HELPER_AVAILABLE = False

# Import the Advanced Grammatical Reformulation Engine (AdvancedPhraseReformulator.py)
try:
    from AdvancedPhraseReformulator import AdvancedPhraseReformulator
    REFORMULATOR_AVAILABLE = True
except ImportError:
    REFORMULATOR_AVAILABLE = False

# Import the Dynamic User Memory and Profile Engine (dynamic_user_memory.py)
try:
    from dynamic_user_memory import DynamicUserMemory
    USER_MEMORY_AVAILABLE = True
except ImportError:
    USER_MEMORY_AVAILABLE = False


# ==============================================================================
# 1. CORE SYSTEM ARCHITECTURE CONFIGURATION
# ==============================================================================

class SystemConfig:
    """
    Consolidated configuration engine managing both structural neural hyperparameters
    and dynamic operational file endpoints.
    """
    def __init__(self):
        # File System Settings
        self.data_file = "big_book.txt"
        self.responses_file = "ollama_responses.txt"
        self.checkpoint_dir = "checkpoints"
        self.model_checkpoint = os.path.join(self.checkpoint_dir, "ai_deep_pytorch.pth")
        self.tokenizer_checkpoint = os.path.join(self.checkpoint_dir, "bpe_tokenizer.json")
        self.fsr_checkpoint_dir = os.path.join(self.checkpoint_dir, "fsr")
        
        # External LLM / Ollama Integrations
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "gemma2:2b"  # Default micro LLM (highly adjustable)
        
        # Neural Network Dimensions
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 4
        self.ff_dim = 1024
        self.dropout = 0.1
        self.max_seq_len = 128  # Target context length for generation
        
        # Optimizer Configuration
        self.batch_size = 16
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.epochs = 5
        self.grad_clip = 1.0
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        # Create checkpoint directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.fsr_checkpoint_dir, exist_ok=True)


# ==============================================================================
# 2. CUSTOM NEURAL MODEL ARCHITECTURE FROM SCRATCH
# ==============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding to provide sequence order parameters to the
    input token embedding maps.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, seq_len, embed_dim]
        return x + self.pe[:, :x.size(1)]


class CustomMultiHeadSelfAttention(nn.Module):
    """
    Clean room implementation of the Causal Multi-Head Self-Attention block
    relying only on low-level matrix multiplication and masking tensors.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by heads count."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection parameters
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Shape transformations: [batch_size, num_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Context vector multiplication
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(context)


class CustomTransformerBlock(nn.Module):
    """
    Structural transformer decoder block executing multi-head attention followed by 
    linear expansions with residual layers and Pre-Layer Normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CustomMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Causal Attention + Residual Layer
        x = x + self.attn(self.ln1(x), mask=mask)
        # Feedforward + Residual Layer
        x = x + self.ff(self.ln2(x))
        return x


class DeepGenerativeTransformer(nn.Module):
    """
    Autonomous generative transformer utilizing custom sinusoidal mappings and BPE tokenization feeds
    to construct, predict, and auto-regressively generate human-like thoughts.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, ff_dim: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embeddings = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.position_embeddings = SinusoidalPositionalEncoding(embed_dim, max_len=max_seq_len + 100)
        
        self.layers = nn.ModuleList([
            CustomTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Structural weight tying between input projections and output head
        self.token_embeddings.embeddings.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.size()
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max target bounds {self.max_seq_len}"
        
        # Build strict Causal Look-ahead Mask
        mask = torch.tril(torch.ones((seq_len, seq_len), device=idx.device)).view(1, 1, seq_len, seq_len)
        
        # Embed inputs
        x = self.token_embeddings(idx)
        x = self.position_embeddings(x)
        
        # Process Layers
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ==============================================================================
# 3. ADVANCED MULTI-FILE SEMANTIC VECTOR DATABASE
# ==============================================================================

class MultiFileSemanticMemory:
    """
    Analyzes local directory environments, reads custom system files (.py, .txt) sémantiquement,
    and indexes them with TF-IDF and Cosine similarity measurements.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=1)
        self.paragraphs: List[str] = []
        self.vectors = None

    def index_target_environment(self, file_paths: List[str]):
        """
        Reads files, chunks code and natural texts sémantiquement,
        and constructs an integrated TF-IDF retrieval database.
        """
        self.paragraphs = []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Split rules based on extensions
                if path.endswith('.py'):
                    lines = content.split('\n')
                    chunk_step = 12  # Pack 12 functional code lines together
                    for i in range(0, len(lines), chunk_step):
                        chunk_text = "\n".join(lines[i:i+chunk_step]).strip()
                        if len(chunk_text) > 35:
                            self.paragraphs.append(f"[Source Code - File: {path} - Lines {i+1}-{i+len(lines[i:i+chunk_step])}]:\n{chunk_text}")
                else:
                    # Default text splitting
                    raw_chunks = content.split('\n\n')
                    for chunk in raw_chunks:
                        cleaned = chunk.strip()
                        if len(cleaned) > 20:
                            self.paragraphs.append(f"[Document Segment - File: {path}]:\n{cleaned}")
            except Exception as e:
                print(f"[Memory Warning] Skipped {path} reading: {e}")

        # Update TF-IDF indices
        if self.paragraphs:
            try:
                self.vectors = self.vectorizer.fit_transform(self.paragraphs)
                print(f"[Semantic Memory] System successfully indexed {len(self.paragraphs)} semantic fragments.")
            except Exception as e:
                print(f"[Memory Error] Vector construction failed: {e}")
                self.vectors = None
        else:
            self.vectors = None

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """Retrieves matching context blocks using cosine similarities."""
        if self.vectors is None or not self.paragraphs:
            return ""
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            best_match_indices = np.argsort(similarities)[::-1][:top_k]
            
            context_blocks = []
            for idx in best_match_indices:
                if similarities[idx] > 0.05:
                    context_blocks.append(self.paragraphs[idx])
            return "\n\n".join(context_blocks)
        except Exception:
            return ""


# ==============================================================================
# 4. MASTER INTERACTIVE SYSTEM CONTROL PIPELINE
# ==============================================================================

class DeepAISystemManager:
    """
    Primary interface controller coordinates standard training loops, custom BPE Tokenizer,
    Multi-file context indexers, local weight caching, and local Ollama interactions.
    Now supports modular integration with the French Sentence Reconstruction (FSR) Library.
    """
    def __init__(self):
        self.config = SystemConfig()
        self.memory = MultiFileSemanticMemory()
        
        # Tracks if model has been successfully optimized/trained
        self.is_model_trained = False
        
        # Build preliminary texts if absent
        self._initialize_core_files()
        
        # Build premium BPE Tokenizer configurations
        self.tokenizer_config = TokenizerConfig(
            vocab_size=1000,  # Expandable target size for robust BPE merging
            min_frequency=1,
            lowercase=True,
            strip_accents=True,
            unicode_normalization="NFKC",
            max_length=self.config.max_seq_len,
            padding=True,
            truncation=True
        )
        self.tokenizer = BytePairEncodingTokenizer(self.tokenizer_config)
        
        # Train and set up tokenizer
        self.train_and_save_tokenizer()
        
        # Instantiate deep neural model
        self.model = DeepGenerativeTransformer(
            vocab_size=self.tokenizer.vocab.size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            ff_dim=self.config.ff_dim,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout
        ).to(self.config.device)
        
        # Initialize French Sentence Reconstruction library properties if available
        self.fsr_active = False
        self.fsr_reconstructor: Optional[FrenchSentenceReconstructor] = None
        if FSR_AVAILABLE:
            self.fsr_reconstructor = FrenchSentenceReconstructor(
                architecture="transformer",
                embed_dim=128,
                hidden_dim=256,
                num_layers=2,
                num_heads=4,
                device=self.config.device
            )
            # Try loading pre-existing checkpoint
            if os.path.exists(os.path.join(self.config.fsr_checkpoint_dir, "model.pt")):
                try:
                    self.fsr_reconstructor = FrenchSentenceReconstructor.load(
                        self.config.fsr_checkpoint_dir, device=self.config.device
                    )
                    self.fsr_active = True
                except Exception as e:
                    print(f"[FSR Manager] Skipped automated loading of saved checkpoint: {e}")

        # Initialize the Advanced Grammatical Reformulation Engine
        self.reformer = None
        if REFORMULATOR_AVAILABLE:
            self.reformer = AdvancedPhraseReformulator(
                max_length=15,
                use_gpu_if_available=True,
                corpus_file_path=self.config.data_file
            )

        # Initialize the Dynamic User Memory and Profile System
        self.user_memory = None
        if USER_MEMORY_AVAILABLE:
            self.user_memory = DynamicUserMemory()

        # Try loading pre-existing weights
        self.load_system_state()
        
        # Index local file tree structures
        self.refresh_memory()

    def _initialize_core_files(self):
        """Builds standard learning documents in the workspace environment."""
        if not os.path.exists(self.config.data_file):
            print(f"[*] Initializing central learning repository: '{self.config.data_file}'...")
            initial_text = (
                "Artificial intelligence utilizes mathematical abstractions to represent language structures.\n"
                "A neural network optimizes numerical weight matrices based on computed losses.\n"
                "Byte-pair encoding decomposes characters iteratively into subword fragments to avoid unknown tokens.\n"
                "Multi-head self-attention relates distinct sequences to build complex semantic representations.\n"
                "By training on local Python codes and documents, the system discovers functional software syntax.\n\n"
                "Google Gemini built the advanced tokenization suite for professional deployment.\n"
                "Local developers run micro LLMs like Gemma locally to support collaborative architectures.\n"
            )
            with open(self.config.data_file, 'w', encoding='utf-8') as f:
                f.write(initial_text)
                
        if not os.path.exists(self.config.responses_file):
            with open(self.config.responses_file, 'w', encoding='utf-8') as f:
                f.write("[Ollama Cooperative Deep Training Responses Log]\n\n")

    def get_local_source_files(self) -> List[str]:
        """Collects all relevant text and Python source files in current workspace folder."""
        targets = [self.config.data_file, self.config.responses_file]
        
        # Append available Python scripts
        for script in glob.glob("*.py"):
            if script not in targets:
                targets.append(script)
                
        # Append available Text documents
        for doc in glob.glob("*.txt"):
            if doc not in targets:
                targets.append(doc)
                
        return list(set(targets))

    def refresh_memory(self):
        """Indexes all discoverable codes and text documentation in the environment."""
        local_files = self.get_local_source_files()
        print(f"[*] Re-indexing Semantic Memory base with local files: {local_files}")
        self.memory.index_target_environment(local_files)

    def train_and_save_tokenizer(self):
        """Trains the premium BPE Tokenizer on all local knowledge bases."""
        sources = [self.config.data_file, self.config.responses_file]
        training_corpus = []
        
        for source in sources:
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                        training_corpus.extend(f.readlines())
                except Exception:
                    pass
                    
        # Filter clean strings
        cleaned_corpus = [line.strip() for line in training_corpus if len(line.strip()) > 3]
        if not cleaned_corpus:
            cleaned_corpus = ["Self-contained default BPE initialization string schema."]
            
        print(f"[*] Training premium Byte-Pair Encoding (BPE) Tokenizer over {len(cleaned_corpus)} text inputs...")
        self.tokenizer.train_from_iterator(cleaned_corpus)
        
        # Force save tokenizer serialization configs
        self.tokenizer.save(self.config.tokenizer_checkpoint)
        print(f"[Tokenizer] BPE Vocabulary built successfully. Active Size: {self.tokenizer.vocab.size} merges.")

    def save_system_state(self):
        """Serializes current neural network weights."""
        payload = {
            'model_weights': self.model.state_dict(),
            'is_model_trained': self.is_model_trained,
            'config_params': {
                'vocab_size': self.tokenizer.vocab.size,
                'embed_dim': self.config.embed_dim,
                'num_heads': self.config.num_heads,
                'num_layers': self.config.num_layers,
                'ff_dim': self.config.ff_dim,
                'max_seq_len': self.config.max_seq_len
            }
        }
        torch.save(payload, self.config.model_checkpoint)
        print(f"[Brain Manager] Serialized model weights safely cached: '{self.config.model_checkpoint}'")

    def load_system_state(self):
        """Loads serialized checkpoint file and updates model structures if present."""
        if os.path.exists(self.config.model_checkpoint):
            try:
                # Load Tokenizer state first
                self.tokenizer.load(self.config.tokenizer_checkpoint)
                
                # Load model weights
                payload = torch.load(self.config.model_checkpoint, map_location=self.config.device)
                
                # Rebuild structure with target parameters
                self.model = DeepGenerativeTransformer(
                    vocab_size=self.tokenizer.vocab.size,
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    num_layers=self.config.num_layers,
                    ff_dim=self.config.ff_dim,
                    max_seq_len=self.config.max_seq_len,
                    dropout=self.config.dropout
                ).to(self.config.device)
                
                self.model.load_state_dict(payload['model_weights'])
                self.is_model_trained = payload.get('is_model_trained', True)
                print(f"[Brain Manager] Successfully restored checkpoint. Vocab Size: {self.tokenizer.vocab.size}")
            except Exception as e:
                print(f"[Warning] Loading checkpoint failed. Starting with empty weights. Detail: {e}")

    def train_on_local_corpus(self, epochs_override: Optional[int] = None):
        """
        Orchestrates an end-to-end Pytorch training loop using the premium TokenizedDataset 
        and DataCollatorWithPadding to adjust generative weights on newly integrated documents.
        """
        print("[Training Pipeline] Compiling all local documents into unified corpus...")
        self.train_and_save_tokenizer()
        
        raw_corpus_list = []
        for path in [self.config.data_file, self.config.responses_file]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        raw_corpus_list.extend([line.strip() for line in lines if len(line.strip()) > 5])
                except Exception:
                    pass
                    
        if not raw_corpus_list:
            print("[Error] No training sequences collected. Populate data files before launching training.")
            return

        # Adapt model output layer dimension dynamically if vocabulary changes occurred
        old_weights = self.model.state_dict()
        self.model = DeepGenerativeTransformer(
            vocab_size=self.tokenizer.vocab.size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            ff_dim=self.config.ff_dim,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout
        ).to(self.config.device)
        
        # Load matched parameters
        new_weights = self.model.state_dict()
        for key, val in old_weights.items():
            if key in new_weights and val.size() == new_weights[key].size():
                new_weights[key].copy_(val)
        self.model.load_state_dict(new_weights)

        # Build PyTorch Dataset & Collators directly from your premium Tokenization Suite
        dataset = TokenizedDataset(raw_corpus_list, self.tokenizer)
        pad_id = self.tokenizer.vocab.lookup_token(self.tokenizer.config.pad_token)
        collator = DataCollatorWithPadding(pad_token_id=pad_id)
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collator)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        
        self.model.train()
        total_epochs = epochs_override if epochs_override is not None else self.config.epochs
        print(f"[Training Pipeline] Optimizing weights on: {self.config.device} over {total_epochs} epochs...")
        
        for epoch in range(total_epochs):
            epoch_loss = 0.0
            steps = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.config.device)
                
                # Check sequence length constraints
                if input_ids.size(1) <= 1:
                    continue
                    
                # Create standard autoregressive target mappings (predict next token)
                x = input_ids[:, :-1]
                y = input_ids[:, 1:]
                
                optimizer.zero_grad()
                logits = self.model(x)
                
                # Compute loss over flattened projection layers
                loss = criterion(logits.reshape(-1, self.tokenizer.vocab.size), y.reshape(-1))
                loss.backward()
                
                # Gradient clipping to prevent parameter explosions
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
                
            avg_loss = epoch_loss / max(1, steps)
            perplexity = math.exp(min(avg_loss, 20))
            print(f"  -> Epoch {epoch+1:02d}/{total_epochs:02d} | Causal Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

        self.is_model_trained = True
        self.save_system_state()
        self.refresh_memory()
        print("[Training Pipeline] Optimization run completed.")

    def generate_autoregressive_thought(self, prompt: str, max_tokens: int = 100, temperature: float = 0.25) -> str:
        """
        Regressively predicts, assembles, and generates character-subword BPE streams 
        based on the retrieved context memories, dynamic user profiles, and generative weights.
        """
        # 1. Update user profile and context history
        context_preamble = ""
        if self.user_memory is not None:
            # Extract names, ages, likes from user prompt
            self.user_memory.extract_profile_facts(prompt)
            # Detect agreement state (yes, no, etc.)
            agreement = self.user_memory.detect_agreement_state(prompt)
            if agreement is not None:
                self.user_memory.update_history("user", f"[User stated {'Affirmative/Yes' if agreement else 'Negative/No'}]")
            else:
                self.user_memory.update_history("user", prompt)
            
            # Fetch structured context preamble
            context_preamble = self.user_memory.get_context_preamble()

        # FALLBACK SAFETY TRIGGER:
        # If the model has not been trained yet, do not output chaotic random chars.
        # Fallback to an optimized local RAG contextual lookup coupled with Markov transition lines!
        if HELPER_AVAILABLE and not self.is_model_trained:
            markov = FailsafeMarkovEngine(order=2)
            
            # Feed documents to the backup model
            for filepath in self.get_local_source_files():
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            markov.fit_text(f.read())
                    except Exception:
                        pass
                        
            # Query semantic database for best context first
            retrieved_passage = self.memory.retrieve_context(prompt, top_k=1)
            seed_words = prompt.split()
            if retrieved_passage:
                clean_passage = re.sub(r"\[Source Code.*?\]", "", retrieved_passage)
                clean_passage = re.sub(r"\[Document Segment.*?\]", "", clean_passage)
                clean_passage = re.sub(r"[^\w\s.,!?]", "", clean_passage)
                passage_words = [w for w in clean_passage.split() if w.strip()]
                if passage_words:
                    seed_words = passage_words[:3] + seed_words
                
            raw_fallback = markov.generate(seed_words=seed_words, max_words=30)
            
            # Apply Output Optimizer
            contraction_handler_inst = FrenchContractionHandler if FSR_AVAILABLE else None
            final_resp = AIOutputOptimizer.optimize_text(raw_fallback, contraction_handler=contraction_handler_inst)
            
            # Update history with fallback response
            if self.user_memory is not None:
                self.user_memory.update_history("assistant", final_resp)
                
            return final_resp

        self.model.eval()
        
        # Retrieve context from local files using Semantic Memory indexer
        retrieved_passage = self.memory.retrieve_context(prompt, top_k=2)
        synthetic_context = ""
        
        # Inject dynamic context preamble (facts + rolling history)
        if context_preamble:
            synthetic_context += context_preamble + "\n"
            
        if retrieved_passage:
            synthetic_context += retrieved_passage + "\n"
        synthetic_context += prompt
        
        # Tokenize prompt utilizing the premium suite BPE mappings
        encoding = self.tokenizer.encode(synthetic_context, add_special_tokens=True)
        input_ids = encoding.ids
        
        # Clip inputs to preserve sequence bounds
        if len(input_ids) > self.config.max_seq_len:
            input_ids = input_ids[-self.config.max_seq_len:]
            
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.config.device)
        generated_indices = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if input_tensor.size(1) > self.config.max_seq_len:
                    context_tensor = input_tensor[:, -self.config.max_seq_len:]
                else:
                    context_tensor = input_tensor
                    
                logits = self.model(context_tensor)
                next_token_logits = logits[0, -1, :]
                
                # Apply advanced SmartSampler from our helper file with stabilized temperature bounds
                if HELPER_AVAILABLE:
                    next_id = SmartSampler.sample_logits(next_token_logits, temperature=temperature, top_k=30, top_p=0.85)
                else:
                    probabilities = F.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
                    next_id = torch.multinomial(probabilities, num_samples=1).item()
                
                # Break generation if target EOS token index is output
                eos_id = self.tokenizer.vocab.lookup_token(self.tokenizer.config.eos_token)
                if next_id == eos_id:
                    break
                    
                generated_indices.append(next_id)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]], device=self.config.device)], dim=1)

        # Decode selected ids back to human text via Premium suite decoding pipeline
        raw_output = self.tokenizer.decode(generated_indices, skip_special_tokens=True)
        
        # Optimize output with helper post-processor
        if HELPER_AVAILABLE:
            contraction_handler_inst = FrenchContractionHandler if FSR_AVAILABLE else None
            final_resp = AIOutputOptimizer.optimize_text(raw_output, contraction_handler=contraction_handler_inst)
        else:
            final_resp = raw_output

        # Update rolling conversation memory
        if self.user_memory is not None:
            self.user_memory.update_history("assistant", final_resp)
            
        return final_resp

    def query_local_ollama(self, system_prompt: str) -> Optional[str]:
        """Performs raw network requests to target Ollama port."""
        payload = {
            "model": self.config.ollama_model,
            "prompt": system_prompt,
            "stream": False
        }
        try:
            res = requests.post(self.config.ollama_url, json=payload, timeout=25)
            if res.status_code == 200:
                return res.json().get("response", "")
        except requests.exceptions.RequestException:
            return None
        return None

    def execute_ollama_collaborative_loop(self, total_prompts: int = 1):
        """
        Cooperative multi-model training routine:
        1. Selects random paragraph chunks from your local indexed codebase or books.
        2. Queries the chosen Ollama model (adjustable) to expand the concept.
        3. Writes the output into 'ollama_responses.txt' and 'big_book.txt'.
        4. Triggers automatic training over newly integrated materials.
        """
        print(f"\n[Ollama Loop] Launching {total_prompts} autonomous generation loop(s) using model: '{self.config.ollama_model}'...")
        if not self.memory.paragraphs:
            print("[Error] No files indexed. Add files or run /status to evaluate directory setup.")
            return

        for step in range(total_prompts):
            print(f"\n--- Processing Cooperative Step {step+1}/{total_prompts} ---")
            seed_passage = random.choice(self.memory.paragraphs)
            print(f"[*] Extracting random memory block as seed: \n    \"{seed_passage[:100]}...\"")
            
            prompt_payload = (
                f"Given this structural seed passage from our internal local base: '{seed_passage}'. "
                "Produce a short, explanatory, clean-room technical summary based on this theme. Keep it concise."
            )
            
            print(f"[*] Calling local Ollama server...")
            response = self.query_local_ollama(prompt_payload)
            
            if not response:
                print("[Error] Failed to communicate with local Ollama server. Check server running state ('ollama serve').")
                return
                
            cleaned_response = response.strip()
            print(f"\n[+] Ollama Response Received:\n{'-'*70}\n{cleaned_response}\n{'-'*70}")
            
            # Save response to primary data book
            with open(self.config.data_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n[Ollama Synthesized Block - Source Model: {self.config.ollama_model}]\n{cleaned_response}")
                
            # Log separate response track in responses_file
            with open(self.config.responses_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== RECOVERY LOG - {self.config.ollama_model} - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{cleaned_response}")
                
            print(f"[+] Saved generated knowledge to '{self.config.data_file}' & '{self.config.responses_file}'.")

        # Refit model immediately on integrated inputs
        print("\n[*] Synchronizing learning parameters over the updated corpus files...")
        self.train_on_local_corpus(epochs_override=2)

    # ==========================================================================
    # FRENCH RECONSTRUCTION (FSR) INTEGRATED METHODS
    # ==========================================================================

    def train_fsr_model(self, epochs: int = 5):
        """
        Builds vocabulary and trains the custom Transformer Seq2Seq Model
        implemented inside the FSR-Lib to handle corrupted French text and contractions.
        """
        if not FSR_AVAILABLE or self.fsr_reconstructor is None:
            print("[Error] FrenchContractionHandler.py not available in the workspace directory.")
            return

        print("[FSR Training] Compiling all local textual lines for French Reconstruction model...")
        raw_sentences = []
        for path in [self.config.data_file, self.config.responses_file]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            cleaned = line.strip()
                            if len(cleaned) > 10:
                                raw_sentences.append(cleaned)
                except Exception:
                    pass

        # Fallback to realistic templates if files are empty
        if len(raw_sentences) < 10:
            print("[FSR Training] Local dataset small. Hydrating with high-fidelity French syntaxes...")
            raw_sentences.extend([
                "L'intelligence artificielle transforme notre monde en profondeur.",
                "Le chat dort paisiblement sur le canapé vert du salon.",
                "Où se trouve la bibliothèque nationale d'informatique ?",
                "C'est une belle journée d'apprentissage pour les réseaux de neurones.",
                "L'arbre devant l'ordinateur portable a perdu toutes ses feuilles.",
                "Je préfère le thé noir d'origine au café au lait chaud.",
                "Le developpement de cette technologie d'attention prend beaucoup de temps."
            ])

        print(f"[FSR Training] Rebuilding FSR Vocabulary using {len(raw_sentences)} sentences...")
        self.fsr_reconstructor.build_vocabulary(raw_sentences, min_freq=1)

        print(f"[FSR Training] Starting French Reconstructor training on: {self.config.device} for {epochs} epochs...")
        self.fsr_reconstructor.train(
            corpus=raw_sentences,
            epochs=epochs,
            batch_size=8 if len(raw_sentences) < 32 else 16,
            learning_rate=0.003,
            val_split=0.1,
            checkpoint_dir=self.config.fsr_checkpoint_dir
        )
        self.fsr_active = True
        self.fsr_reconstructor.save(self.config.fsr_checkpoint_dir)
        print(f"[FSR Training] Successfully completed and saved FSR parameters to '{self.config.fsr_checkpoint_dir}'.")

    def reconstruct_french_input(self, corrupted_sentence: str) -> str:
        """
        Decodes a corrupted French sentence utilizing FSR-Lib contraction rules and
        beam search sequence matching.
        """
        if not FSR_AVAILABLE or self.fsr_reconstructor is None:
            return "[Error] French Sentence Reconstruction module is not imported."
        if not self.fsr_active:
            return "[Warning] Reconstructor is not trained yet. Run '/train_fsr' first."
        try:
            return self.fsr_reconstructor.reconstruct(corrupted_sentence, method="beam")
        except Exception as e:
            return f"[FSR Reconstruction Failure] : {e}"


# ==============================================================================
# 5. CORE CONSOLE INTERFACE ENGINE
# ==============================================================================

def main():
    # Instantiate management engine
    system = DeepAISystemManager()
    
    print("\n" + "="*80)
    print("                 HYBRID PYTORCH GENERATIVE CORE ACTIVE")
    print("      [Tokenization: BPE Suite • Engine: Custom Transformer Decoder]")
    if FSR_AVAILABLE:
        print("      [French Reconstruction Module (FSR-Lib): Successfully Imported]")
    else:
        print("      [French Reconstruction Module (FSR-Lib): Not Available]")
    if USER_MEMORY_AVAILABLE:
        print("      [Dynamic User Profile & Context History Memory: Successfully Active]")
    else:
        print("      [Dynamic User Profile & Context History Memory: Import Failed]")
    print("="*80)
    print(" Core System Commands:")
    print("   /train <epochs>   - Re-optimize model weights on all local codebase files")
    print("   /model <name>     - Set active local Ollama target (e.g. /model llama3)")
    print("   /gemma <count>    - Trigger cooperative Ollama processing steps & refit loops")
    print("   /status           - Inspect vocabulary sizes, indexed files, and cuda details")
    print("   /save             - Export current network parameter tensors to checkpoints")
    print("   /load             - Reload states from ai_deep_pytorch.pth checkpoint")
    if FSR_AVAILABLE:
        print("   /train_fsr <ep>   - Train custom French Sentence Reconstruction model on your book")
        print("   /reconstruct <tx> - Denoise, reconstruct, and fix typos/contractions in French text")
    print("   /clear            - Flush CLI terminal screen output")
    print("   /exit             - Safely close AI pipeline structures")
    print("="*80 + "\n")
    
    while True:
        try:
            user_input = input("\nAI-CLI >>> ").strip()
            if not user_input:
                continue
                
            if user_input.startswith("/"):
                tokens = user_input.split(" ")
                cmd = tokens[0].lower()
                
                if cmd == "/exit":
                    print("[*] Terminating processes. Saving brain checkpoints...")
                    system.save_system_state()
                    break
                    
                elif cmd == "/train":
                    epochs = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else None
                    system.train_on_local_corpus(epochs_override=epochs)
                    
                elif cmd == "/model":
                    if len(tokens) > 1:
                        target_model = tokens[1]
                        system.config.ollama_model = target_model
                        print(f"[Config] Updated active local Ollama model identifier to: '{target_model}'")
                    else:
                        print("[!] Please provide a valid model tag. Example: /model gemma2:2b")
                        
                elif cmd == "/gemma":
                    count = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 1
                    system.execute_ollama_collaborative_loop(total_prompts=count)
                    
                elif cmd == "/train_fsr":
                    if FSR_AVAILABLE:
                        epochs = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 5
                        system.train_fsr_model(epochs=epochs)
                    else:
                        print("[!] FrenchSentenceReconstructor file is not present in workspace.")

                elif cmd == "/reconstruct":
                    if FSR_AVAILABLE:
                        if len(tokens) > 1:
                            text_to_fix = " ".join(tokens[1:])
                            reconstructed_res = system.reconstruct_french_input(text_to_fix)
                            print(f"\n[Original Corrupted Input]:  {text_to_fix}")
                            print(f"[FSR Corrected Prediction]:  {reconstructed_res}")
                        else:
                            print("[!] Please provide the text to reconstruct. Example: /reconstruct l'intelgence artficle")
                    else:
                        print("[!] FrenchSentenceReconstructor file is not present in workspace.")

                elif cmd == "/status":
                    print("\n" + "-"*40 + "\n--- Pipeline Diagnostics ---")
                    print(f"  Execution Device  : {system.config.device.upper()}")
                    print(f"  BPE Vocab Size    : {system.tokenizer.vocab.size} unique merged tokens")
                    print(f"  Embedding Dim     : {system.config.embed_dim}")
                    print(f"  Encoder Channels  : {system.config.num_layers} custom blocks")
                    print(f"  Ollama Model      : {system.config.ollama_model}")
                    print(f"  FSR-Lib Status    : {'Active & Loaded' if system.fsr_active else 'Inactive (Needs /train_fsr)' if FSR_AVAILABLE else 'Import Failed'}")
                    print(f"  User Memory Status: {'Active & Persisted' if system.user_memory else 'Inactive'}")
                    print("  Memory Source Tree:")
                    for path in system.get_local_source_files():
                        if os.path.exists(path):
                            print(f"    - {path} ({os.path.getsize(path)} bytes)")
                    print("-" * 40)
                    
                elif cmd == "/save":
                    system.save_system_state()
                    if system.fsr_active and system.fsr_reconstructor:
                        system.fsr_reconstructor.save(system.config.fsr_checkpoint_dir)
                    
                elif cmd == "/load":
                    system.load_system_state()
                    system.refresh_memory()
                    if FSR_AVAILABLE and system.fsr_reconstructor:
                        try:
                            system.fsr_reconstructor = FrenchSentenceReconstructor.load(
                                system.config.fsr_checkpoint_dir, device=system.config.device
                            )
                            system.fsr_active = True
                            print("[FSR Manager] Restored FSR network weights successfully.")
                        except Exception as e:
                            print(f"[FSR Manager] Restoring FSR weights failed: {e}")
                    
                elif cmd == "/clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                else:
                    print("[!] Unrecognized systemic command instruction. Run /status to verify targets.")
            else:
                # Run autoregressive thought synthesis with low-temperature safety (0.25)
                print("\n[*] Thinking...")
                reconstructed_output = system.generate_autoregressive_thought(user_input, max_tokens=100, temperature=0.25)
                print(f"\n[Generated Thought Response]:\n{reconstructed_output}")
                
        except KeyboardInterrupt:
            print("\n[*] Interruption detected. Safely stopping core management systems...")
            sys.exit(0)
        except Exception as e:
            print(f"[Core Exception Caught] : {e}")

if __name__ == "__main__":
    main()