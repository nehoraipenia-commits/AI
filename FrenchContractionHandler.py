"""
================================================================================
          FRENCH SENTENCE RECONSTRUCTION LIBRARY (FSR-Lib)
================================================================================
A comprehensive, self-contained, production-grade deep learning framework
developed in PyTorch and NumPy for correcting, denosing, and reconstructing
corrupted French sentences (typos, word-drops, ordering, grammar, and masks).

Architectures Implemented:
1. Custom Seq2Seq LSTM with Bahdanau Attention (from scratch)
2. Custom Full Transformer Encoder-Decoder Network (from scratch)

Language: English (Code, Comments, API, Logs)
Target Domain: French Sentences (Accents, Contractions, Syntactic Patterns)
"""

import os
import re
import math
import time
import random
import json
from collections import Counter, namedtuple
from typing import List, Tuple, Dict, Set, Union, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Global Seed Configuration
def set_seed(seed: int = 42):
    """Sets seed for reproducibility across CPU, GPU, and libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


# =====================================================================
# 1. FRENCH LINGUISTICS & TEXT PREPROCESSING MODULE
# =====================================================================

class FrenchContractionHandler:
    """
    Handles specialized French contractions and clitics (e.g., d', l', qu', s', m', t', -je, -il).
    Separates prefixes or suffixes to allow clean word-level or subword tokenization.
    """
    CONTRACTION_PREFIXES = [
        r"\b[lL]'", r"\b[dD]'", r"\b[jJ]'", r"\b[nN]'", r"\b[sS]'", 
        r"\b[mM]'", r"\b[tT]'", r"\b[cC]'", r"\b[qQ]u'", r"\b[pP]uqu'"
    ]
    HYPHENATED_SUFFIXES = [
        r"-je\b", r"-tu\b", r"-il\b", r"-elle\b", r"-on\b", r"-nous\b", 
        r"-vous\b", r"-ils\b", r"-elles\b", r"-ci\b", r"-là\b", r"-en\b", r"-y\b"
    ]

    @classmethod
    def split_contractions(cls, text: str) -> str:
        """
        Splits French contractions into individual token components.
        Example: "L'intelligence artificielle" -> "L' intelligence artificielle"
        """
        # Prefix contractions splitting (e.g., l'arbre -> l' arbre)
        for prefix in cls.CONTRACTION_PREFIXES:
            text = re.sub(prefix, lambda m: m.group(0) + " ", text)
            
        # Suffix hyphenated forms splitting (e.g., dit-il -> dit -il)
        for suffix in cls.HYPHENATED_SUFFIXES:
            text = re.sub(suffix, lambda m: " " + m.group(0), text)
            
        # Standardize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def merge_contractions(cls, tokens: List[str]) -> str:
        """
        Reconstructs the original French contractive spacing from token sequences.
        Example: ["l'", "intelligence"] -> "l'intelligence"
        """
        sentence = " ".join(tokens)
        # Remove spaces after contractions
        sentence = re.sub(r"\b([lLdDjJnNsSmMtTcC]|qu|puqu)'\s+", r"\1'", sentence)
        sentence = re.sub(r"\b([lLdDjJnNsSmMtTcC]|QU|PUQU)'\s+", r"\1'", sentence, flags=re.IGNORECASE)
        # Remove spaces before hyphenated suffixes
        sentence = re.sub(r"\s+(-(je|tu|il|elle|on|nous|vous|ils|elles|ci|là|en|y))\b", r"\1", sentence)
        return sentence


class FrenchSentenceTokenizer:
    """
    A robust, deterministic tokenizer designed for French text normalization,
    contraction splitting, punctuation handling, and case configurations.
    """
    PUNCTUATION_PATTERN = re.compile(r"([.,!?;:«»()\"\[\]])")

    def __init__(self, lowercase: bool = True, split_french_contractions: bool = True):
        self.lowercase = lowercase
        self.split_french_contractions = split_french_contractions

    def tokenize(self, text: str) -> List[str]:
        """Converts a raw French sentence into a sequence of string tokens."""
        if self.lowercase:
            text = text.lower()
            
        if self.split_french_contractions:
            text = FrenchContractionHandler.split_contractions(text)
            
        # Pad punctuation characters with spaces
        text = self.PUNCTUATION_PATTERN.sub(r" \1 ", text)
        
        # Split by white spaces and filter out empty items
        tokens = [t.strip() for t in text.split(" ") if t.strip()]
        return tokens

    def untokenize(self, tokens: List[str]) -> str:
        """Converts a sequence of string tokens back into a readable French sentence."""
        # Join tokens and perform cleaning
        text = " ".join(tokens)
        
        # Clean spacing around common punctuation
        text = re.sub(r"\s+([.,!?;:»\])])", r"\1", text)
        text = re.sub(r"([«\[(\"])\s+", r"\1", text)
        
        # Handle French spacing rules for high double-punctuation (?, !, ;, :)
        text = re.sub(r"\s+([?!;:«»])", r" \1", text) 
        
        # Re-merge contractions if needed
        if self.split_french_contractions:
            text = FrenchContractionHandler.merge_contractions(text.split())
            
        return re.sub(r'\s+', ' ', text).strip()


class Vocabulary:
    """
    Manages the mapping between sentence-level string tokens and numerical indices.
    Supports special tokens, frequency-cutoff thresholding, and serialization.
    """
    PAD_TOKEN = "<PAD>"  # Index 0
    SOS_TOKEN = "<SOS>"  # Index 1
    EOS_TOKEN = "<EOS>"  # Index 2
    UNK_TOKEN = "<UNK>"  # Index 3
    MASK_TOKEN = "<mask>" # Index 4

    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_freqs: Counter = Counter()
        
        # Pre-populate special tokens
        for token in self.SPECIAL_TOKENS:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """Adds a token to the vocabulary mapping structures."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            return idx
        return self.token2idx[token]

    def build_vocab(self, tokenized_sentences: List[List[str]]):
        """Constructs mappings using tokenized text data and min_freq threshold constraints."""
        for sentence in tokenized_sentences:
            for token in sentence:
                self.token_freqs[token] += 1
                
        for token, freq in self.token_freqs.items():
            if freq >= self.min_freq:
                self.add_token(token)
                
        print(f"Vocabulary successfully built. Total Tokens: {len(self.token2idx)} (Cutoff min_freq={self.min_freq})")

    def __len__(self) -> int:
        return len(self.token2idx)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    @property
    def mask_idx(self) -> int:
        return self.token2idx[self.MASK_TOKEN]

    def encode(self, tokens: List[str], add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Transforms string tokens into a list of vocabulary-mapped indices."""
        indices = []
        if add_sos:
            indices.append(self.sos_idx)
        for t in tokens:
            indices.append(self.token2idx.get(t, self.unk_idx))
        if add_eos:
            indices.append(self.eos_idx)
        return indices

    def decode(self, indices: List[int], skip_specials: bool = True) -> List[str]:
        """Transforms mapped indices back into string tokens."""
        tokens = []
        for idx in indices:
            # Handle PyTorch items
            if hasattr(idx, 'item'):
                idx = idx.item()
            token = self.idx2token.get(idx, self.UNK_TOKEN)
            if skip_specials and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    def save_to_file(self, path: str):
        """Serializes current vocabulary parameters into a JSON file."""
        state = {
            "token2idx": self.token2idx,
            "idx2token": {str(k): v for k, v in self.idx2token.items()},
            "min_freq": self.min_freq
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> "Vocabulary":
        """Reconstructs a Vocabulary instance from a serialized JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        vocab = cls(min_freq=state["min_freq"])
        vocab.token2idx = state["token2idx"]
        vocab.idx2token = {int(k): v for k, v in state["idx2token"].items()}
        return vocab


# =====================================================================
# 2. HIGH-FIDELITY SENTENCE CORRUPTION GENERATOR
# =====================================================================

class SentenceCorruptor:
    """
    A multi-mode synthetic noise injection pipeline designed specifically 
    for French text. Simulates common grammatical, spelling, transcription, 
    and typing errors at both the word and character levels.
    """
    def __init__(self, vocab: Vocabulary, error_rate: float = 0.2):
        self.vocab = vocab
        self.error_rate = error_rate
        # Basic character layouts to simulate nearby typing mistakes (AZERTY layout)
        self.azerty_near_keys = {
            'a': 'zqs', 'z': 'aesd', 'e': 'zrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj',
            'u': 'yijhk', 'i': 'uokjl', 'o': 'iplkm', 'p': 'o', 'q': 'asdw', 's': 'qezdxw',
            'd': 'srzfcx', 'f': 'dtegcv', 'g': 'rfyhvb', 'h': 'gtujnb', 'j': 'hyikn',
            'k': 'jilm', 'l': 'kopm', 'm': 'l', 'w': 'qxc', 'x': 'wsdc', 'c': 'xdfv',
            'v': 'cfgb', 'b': 'vghn', 'n': 'bhj', 'é': 'e', 'è': 'e', 'à': 'a', 'ç': 'c'
        }

    def _corrupt_characters(self, word: str, char_error_prob: float = 0.1) -> str:
        """Injects typographical/orthographical noise inside an individual word string."""
        if len(word) <= 1 or word.startswith("<") or word == self.vocab.MASK_TOKEN:
            return word
            
        chars = list(word)
        corrupted_chars = []
        i = 0
        while i < len(chars):
            if random.random() < char_error_prob:
                corruption_type = random.choice(["swap", "delete", "substitute", "insert"])
                
                if corruption_type == "swap" and i < len(chars) - 1:
                    # Character transposition
                    corrupted_chars.append(chars[i+1])
                    corrupted_chars.append(chars[i])
                    i += 2
                    continue
                elif corruption_type == "delete":
                    # Character omission
                    i += 1
                    continue
                elif corruption_type == "substitute":
                    # Typing nearby key substitution
                    c = chars[i].lower()
                    if c in self.azerty_near_keys:
                        corrupted_chars.append(random.choice(self.azerty_near_keys[c]))
                    else:
                        corrupted_chars.append(random.choice("abcdefghijklmnopqrstuvwxyz"))
                    i += 1
                    continue
                elif corruption_type == "insert":
                    # Double key typing or nearby insertion
                    corrupted_chars.append(chars[i])
                    c = chars[i].lower()
                    if c in self.azerty_near_keys:
                        corrupted_chars.append(random.choice(self.azerty_near_keys[c]))
                    else:
                        corrupted_chars.append(chars[i])
                    i += 1
                    continue
            
            corrupted_chars.append(chars[i])
            i += 1
            
        return "".join(corrupted_chars) if corrupted_chars else word

    def corrupt_sentence(self, tokens: List[str]) -> List[str]:
        """
        Applies grammatical and layout corruptions onto a clean sentence token sequence.
        Performs word dropping, swapping, masking, insertion, and token typos.
        """
        if not tokens:
            return []
            
        corrupted_tokens = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Skip special elements
            if token in self.vocab.SPECIAL_TOKENS:
                corrupted_tokens.append(token)
                i += 1
                continue
                
            # Random drop / omission error
            if random.random() < (self.error_rate * 0.4):
                i += 1
                continue
                
            # Random masking (useful for Cloze test/completion style training)
            if random.random() < (self.error_rate * 0.3):
                corrupted_tokens.append(self.vocab.MASK_TOKEN)
                i += 1
                continue
                
            # Random adjacent word swapping
            if random.random() < (self.error_rate * 0.2) and i < len(tokens) - 1:
                next_token = tokens[i+1]
                # Apply optional typos to swapped tokens
                t1 = self._corrupt_characters(next_token, char_error_prob=0.08)
                t2 = self._corrupt_characters(token, char_error_prob=0.08)
                corrupted_tokens.append(t1)
                corrupted_tokens.append(t2)
                i += 2
                continue
                
            # Random word insertion
            if random.random() < (self.error_rate * 0.15):
                # Choose random word from vocabulary keys or common French filler
                fillers = ["le", "un", "de", "ce", "que", "en", "pas", "y"]
                random_filler = random.choice(fillers)
                corrupted_tokens.append(random_filler)
                
            # Character-level spelling errors within current word
            corrupted_word = self._corrupt_characters(token, char_error_prob=0.15)
            corrupted_tokens.append(corrupted_word)
            i += 1
            
        # Guarantee we don't return an entirely empty sentence
        if not corrupted_tokens:
            corrupted_tokens = [self.vocab.MASK_TOKEN]
            
        return corrupted_tokens


# =====================================================================
# 3. PYTORCH DATA PIPELINE
# =====================================================================

class FrenchReconstructionDataset(Dataset):
    """
    Sentence-level dataset returning tokenized, aligned pairs 
    of (corrupted_indices, original_indices) for model optimization.
    """
    def __init__(self, raw_sentences: List[str], tokenizer: FrenchSentenceTokenizer, 
                 vocab: Vocabulary, corruptor: SentenceCorruptor, max_seq_len: int = 64):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.corruptor = corruptor
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[List[str], List[str]]] = []
        
        # Pre-process raw sentences
        for sentence in raw_sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            clean_tokens = self.tokenizer.tokenize(cleaned)
            if len(clean_tokens) > 0 and len(clean_tokens) <= self.max_seq_len - 2:
                self.samples.append(clean_tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clean ground truth tokens
        tgt_tokens = self.samples[idx]
        
        # Inject custom corruptions
        src_tokens = self.corruptor.corrupt_sentence(tgt_tokens)
        
        # Map tokens to sequence tensors with boundary markers
        src_indices = self.vocab.encode(src_tokens, add_sos=False, add_eos=True)
        tgt_indices = self.vocab.encode(tgt_tokens, add_sos=True, add_eos=True)
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)


class DynamicPaddingCollate:
    """Collate utility to perform batch-level dynamic sequence padding."""
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)
        
        # Collate source and targets separately, applying PAD tokens
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_idx)
        
        return src_padded, tgt_padded


# =====================================================================
# 4. CUSTOM NEURAL NETWORKS (FROM SCRATCH)
# =====================================================================

# ---------------------------------------------------------------------
# ARCHITECTURE A: SEQ2SEQ WITH BAHDANAU ATTENTION
# ---------------------------------------------------------------------

class RNNAttentionEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for contextual sequence modeling.
    Outputs continuous hidden state representations for sequence sequences.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Bidirectional LSTM mapping
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, 
                            dropout=dropout if num_layers > 1 else 0.0)
        
        # Dimensionality matching linear layer for cell and hidden transfers
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [Batch, SeqLen]
        embedded = self.embedding(x)  # [Batch, SeqLen, EmbedDim]
        outputs, (h_n, c_n) = self.lstm(embedded)  # outputs: [Batch, SeqLen, HiddenDim * 2]
        
        # Resolve bidirectional states for last layer and project to single hidden dimension
        # Forward states represent index -2, Backward states represent index -1
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        c_forward = c_n[-2, :, :]
        c_backward = c_n[-1, :, :]
        
        # Concatenate forward and backward vectors
        h_combined = torch.cat((h_forward, h_backward), dim=1) # [Batch, HiddenDim * 2]
        c_combined = torch.cat((c_forward, c_backward), dim=1) # [Batch, HiddenDim * 2]
        
        # Project states back to hidden dimensions for Decoder initialization
        h_projected = torch.tanh(self.fc_hidden(h_combined)).unsqueeze(0) # [1, Batch, HiddenDim]
        c_projected = torch.tanh(self.fc_cell(c_combined)).unsqueeze(0)   # [1, Batch, HiddenDim]
        
        return outputs, (h_projected, c_projected)


class BahdanauAttention(nn.Module):
    """
    Additive Attention Module (Bahdanau alignment).
    Calculates weights using a multi-layer perceptron over the target query and source context.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_values = nn.Linear(hidden_dim * 2, hidden_dim)
        self.V_score = nn.Linear(hidden_dim, 1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # query: [1, Batch, HiddenDim] (decoder hidden state)
        # values: [Batch, SeqLen, HiddenDim * 2] (encoder outputs)
        
        # Transform states
        query = query.transpose(0, 1)  # [Batch, 1, HiddenDim]
        seq_len = values.size(1)
        
        query_projected = self.W_query(query)  # [Batch, 1, HiddenDim]
        query_expanded = query_projected.repeat(1, seq_len, 1) # [Batch, SeqLen, HiddenDim]
        
        values_projected = self.W_values(values)  # [Batch, SeqLen, HiddenDim]
        
        # Add values and calculate activation score
        scores = torch.tanh(query_expanded + values_projected)  # [Batch, SeqLen, HiddenDim]
        scores = self.V_score(scores).squeeze(2)  # [Batch, SeqLen]
        
        # Return softmax alignment probability distribution
        return F.softmax(scores, dim=1).unsqueeze(1)  # [Batch, 1, SeqLen]


class RNNAttentionDecoder(nn.Module):
    """
    LSTM Decoder module with Bahdanau attention calculations
    implemented over the complete source memory matrix.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        
        # Decoder input incorporates embedded features and alignment context vectors
        self.lstm = nn.LSTM(embed_dim + (hidden_dim * 2), hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        # Multi-layer output mapping
        self.out = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [Batch, 1] (Single step tokens)
        # hidden: ([1, Batch, HiddenDim], [1, Batch, HiddenDim])
        # encoder_outputs: [Batch, SeqLen, HiddenDim * 2]
        
        embedded = self.embedding(x)  # [Batch, 1, EmbedDim]
        
        # Retrieve context matching matrix
        attn_weights = self.attention(hidden[0], encoder_outputs) # [Batch, 1, SeqLen]
        context = torch.bmm(attn_weights, encoder_outputs)        # [Batch, 1, HiddenDim * 2]
        
        # Build composite inputs and propagate through LSTM cell
        rnn_input = torch.cat((embedded, context), dim=2)  # [Batch, 1, EmbedDim + HiddenDim * 2]
        output, hidden = self.lstm(rnn_input, hidden)     # output: [Batch, 1, HiddenDim]
        
        # Project representation to final vocabulary index distributions
        output = output.squeeze(1)     # [Batch, HiddenDim]
        context = context.squeeze(1)   # [Batch, HiddenDim * 2]
        embedded = embedded.squeeze(1) # [Batch, EmbedDim]
        
        combined = torch.cat((output, context, embedded), dim=1) # [Batch, HiddenDim * 3 + EmbedDim]
        prediction = self.out(combined)  # [Batch, VocabSize]
        
        return prediction, hidden


class RNNAttentionSeq2Seq(nn.Module):
    """Seq2Seq Wrapper for RNN models."""
    def __init__(self, encoder: RNNAttentionEncoder, decoder: RNNAttentionDecoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        enc_outputs, hidden = self.encoder(src)

        # SOS tokens initialization for decoder
        dec_input = tgt[:, 0].unsqueeze(1) # [Batch, 1]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(dec_input, hidden, enc_outputs)
            outputs[:, t, :] = output
            
            # Determine teacher forcing strategy
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top_pred = output.argmax(1).unsqueeze(1)
            dec_input = tgt[:, t].unsqueeze(1) if use_teacher_forcing else top_pred

        return outputs


# ---------------------------------------------------------------------
# ARCHITECTURE B: FULL TRANSFORMER ENCODER-DECODER (FROM SCRATCH)
# ---------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for sequence layout awareness."""
    def __init__(self, embed_dim: int, max_len: int = 250, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, MaxLen, EmbedDim]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, EmbedDim]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Custom implementation of Scaled Dot-Product Multi-Head Attention."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimensions must be divisible by num_heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Shapes: [Batch, SeqLen, EmbedDim]
        batch_size = query.size(0)

        # Apply projections and reshape to split across attention heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Reshaped to: [Batch, Heads, SeqLen, HeadDim]

        # Calculate energy scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # [Batch, Heads, Q_Len, K_Len]

        if mask is not None:
            # Mask is broadcastable: [Batch, 1, 1, K_Len] or [Batch, 1, Q_Len, K_Len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_map = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_map, V) # [Batch, Heads, Q_Len, HeadDim]

        # Merge heads together
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim) # [Batch, Q_Len, EmbedDim]
        return self.fc_out(out)


class PositionwiseFeedForward(nn.Module):
    """Standard Feed-Forward Network Layer for Transformer Blocks."""
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder block."""
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = PositionwiseFeedForward(embed_dim, ff_hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Attention sublayer
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Feed-forward sublayer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class TransformerDecoderLayer(nn.Module):
    """Standard Transformer Decoder block."""
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = PositionwiseFeedForward(embed_dim, ff_hidden_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_outputs: torch.Tensor, 
                self_mask: torch.Tensor, cross_mask: torch.Tensor) -> torch.Tensor:
        # Decoder self-attention
        self_attn_out = self.self_mha(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        
        # Cross-attention over encoder representation
        cross_attn_out = self.cross_mha(x, enc_outputs, enc_outputs, cross_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        
        # Feed-forward sublayer
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


class TransformerSeq2Seq(nn.Module):
    """
    A full Transformer Encoder-Decoder model built entirely from scratch, 
    complete with custom padding and causally masked attention logic.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 3, ff_hidden_dim: int = 256, dropout: float = 0.1, 
                 pad_idx: int = 0, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.pad_idx = pad_idx
        self.device = device
        
        # Embeddings & Positional Encodings
        self.src_embedding = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Stacked blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # src: [Batch, Src_Len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2) # [Batch, 1, 1, Src_Len]
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # tgt: [Batch, Tgt_Len]
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2) # [Batch, 1, 1, Tgt_Len]
        
        # Generate triangular causally-masked sequence matrix
        tgt_len = tgt.size(1)
        sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool() # [Tgt_Len, Tgt_Len]
        
        # Combine both padding mask and causal sequence bounds
        tgt_mask = tgt_pad_mask & sub_mask.unsqueeze(0).unsqueeze(1) # [Batch, 1, Tgt_Len, Tgt_Len]
        return tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.src_embedding(src))
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt: torch.Tensor, enc_outputs: torch.Tensor, 
               tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.tgt_embedding(tgt))
        x = self.pos_encoder(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_outputs, tgt_mask, src_mask)
        return self.fc_out(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        # Note: teacher_forcing_ratio parameter is unused here to preserve compatibility 
        # with standard RNN execution loops (Transformers use full sequence training concurrently).
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_outputs = self.encode(src, src_mask)
        outputs = self.decode(tgt, enc_outputs, tgt_mask, src_mask)
        return outputs


# =====================================================================
# 5. ADVANCED INFERENCE DECODERS
# =====================================================================

class BeamSearchDecoder:
    """
    Advanced Beam Search Decoder with length-penalized sequence searches.
    Compatible with both Seq2Seq RNNs and Transformer network modules.
    """
    BeamCandidate = namedtuple('BeamCandidate', ['sequence', 'score', 'hidden_state'])

    def __init__(self, model: nn.Module, vocab: Vocabulary, beam_size: int = 5, 
                 max_len: int = 64, length_penalty_alpha: float = 0.6):
        self.model = model
        self.vocab = vocab
        self.beam_size = beam_size
        self.max_len = max_len
        self.alpha = length_penalty_alpha
        self.device = model.device

    def _get_length_penalty(self, length: int) -> float:
        """Applies sequence length penalty scaling to scores."""
        return ((5.0 + length) / 6.0) ** self.alpha

    @torch.no_grad()
    def decode_rnn(self, src_tensor: torch.Tensor) -> List[int]:
        """Performs search execution loop over our Seq2Seq RNN model."""
        self.model.eval()
        enc_outputs, rnn_hidden = self.model.encoder(src_tensor)
        
        # Start state initialization
        initial_candidate = self.BeamCandidate(
            sequence=[self.vocab.sos_idx],
            score=0.0,
            hidden_state=rnn_hidden
        )
        
        active_beams = [initial_candidate]
        completed_beams = []

        for step in range(self.max_len):
            new_candidates = []
            
            for beam in active_beams:
                if beam.sequence[-1] == self.vocab.eos_idx:
                    completed_beams.append(beam)
                    continue
                
                # Setup decoder input vector for step
                step_input = torch.tensor([[beam.sequence[-1]]], dtype=torch.long, device=self.device)
                logits, next_hidden = self.model.decoder(step_input, beam.hidden_state, enc_outputs)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [VocabSize]
                
                # Fetch top prediction scores
                val, idx = torch.topk(log_probs, self.beam_size)
                for score_val, token_idx in zip(val, idx):
                    new_seq = beam.sequence + [token_idx.item()]
                    new_score = beam.score + score_val.item()
                    
                    new_candidates.append(self.BeamCandidate(
                        sequence=new_seq,
                        score=new_score,
                        hidden_state=next_hidden
                    ))
            
            if not new_candidates:
                break
                
            # Select overall top beam_size elements
            new_candidates = sorted(new_candidates, key=lambda c: c.score / self._get_length_penalty(len(c.sequence)), reverse=True)
            active_beams = new_candidates[:self.beam_size]
            
            # Stop if all candidate search patterns are completed
            if all(b.sequence[-1] == self.vocab.eos_idx for b in active_beams):
                break

        # Consolidate candidate lists
        all_results = completed_beams + active_beams
        all_results = sorted(all_results, key=lambda c: c.score / self._get_length_penalty(len(c.sequence)), reverse=True)
        
        # Return sequence matching the highest overall penalized log probability score
        return all_results[0].sequence

    @torch.no_grad()
    def decode_transformer(self, src_tensor: torch.Tensor) -> List[int]:
        """Performs search execution loop over our Transformer seq2seq model."""
        self.model.eval()
        src_mask = self.model.make_src_mask(src_tensor)
        enc_outputs = self.model.encode(src_tensor, src_mask)
        
        initial_candidate = self.BeamCandidate(
            sequence=[self.vocab.sos_idx],
            score=0.0,
            hidden_state=None
        )
        
        active_beams = [initial_candidate]
        completed_beams = []

        for step in range(self.max_len):
            new_candidates = []
            
            for beam in active_beams:
                if beam.sequence[-1] == self.vocab.eos_idx:
                    completed_beams.append(beam)
                    continue
                
                # Generate sequence tensor
                tgt_tensor = torch.tensor([beam.sequence], dtype=torch.long, device=self.device)
                tgt_mask = self.model.make_tgt_mask(tgt_tensor)
                
                # Forward decode
                logits = self.model.decode(tgt_tensor, enc_outputs, tgt_mask, src_mask)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # Fetch log-probabilities for last step
                
                val, idx = torch.topk(log_probs, self.beam_size)
                for score_val, token_idx in zip(val, idx):
                    new_seq = beam.sequence + [token_idx.item()]
                    new_score = beam.score + score_val.item()
                    
                    new_candidates.append(self.BeamCandidate(
                        sequence=new_seq,
                        score=new_score,
                        hidden_state=None
                    ))
            
            if not new_candidates:
                break
                
            new_candidates = sorted(new_candidates, key=lambda c: c.score / self._get_length_penalty(len(c.sequence)), reverse=True)
            active_beams = new_candidates[:self.beam_size]
            
            if all(b.sequence[-1] == self.vocab.eos_idx for b in active_beams):
                break

        all_results = completed_beams + active_beams
        all_results = sorted(all_results, key=lambda c: c.score / self._get_length_penalty(len(c.sequence)), reverse=True)
        return all_results[0].sequence


# =====================================================================
# 6. METRICS & PERFORMANCE EVALUATION SUITE
# =====================================================================

class MetricSuite:
    """Provides standard metric scoring functions for sentence reconstruction tasks."""
    
    @staticmethod
    def calculate_levenshtein(seq1: List[str], seq2: List[str]) -> int:
        """Computes Levenshtein edit distance between token sequences."""
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = min(
                        dp[i-1, j] + 1,    # Deletion
                        dp[i, j-1] + 1,    # Insertion
                        dp[i-1, j-1] + 1   # Substitution
                    )
        return dp[m, n]

    @classmethod
    def word_error_rate(cls, hypothesis: List[str], reference: List[str]) -> float:
        """Calculates Word Error Rate (WER) using edit distance normalizations."""
        if not reference:
            return 1.0 if hypothesis else 0.0
        edit_dist = cls.calculate_levenshtein(hypothesis, reference)
        return float(edit_dist) / len(reference)

    @staticmethod
    def sentence_bleu(hypothesis: List[str], reference: List[str], weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> float:
        """
        Calculates a clean 4-gram Sentence BLEU score.
        Includes clipping, geometric mean, and brevity penalties.
        """
        hyp_len = len(hypothesis)
        ref_len = len(reference)
        
        if hyp_len == 0:
            return 0.0

        # Brevity Penalty Calculation
        brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))

        precisions = []
        for n in range(1, len(weights) + 1):
            # Form n-grams
            hyp_ngrams = [tuple(hypothesis[i:i+n]) for i in range(hyp_len - n + 1)]
            ref_ngrams = [tuple(reference[i:i+n]) for i in range(ref_len - n + 1)]
            
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
                
            hyp_counter = Counter(hyp_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            # Count shared n-grams (clipping overlap)
            overlap = 0
            for ngram, count in hyp_counter.items():
                overlap += min(count, ref_counter.get(ngram, 0))
                
            p_n = overlap / len(hyp_ngrams)
            precisions.append(p_n)

        # Apply geometric mean with standard smoothing (using pseudo counts if 0)
        smoothed_scores = []
        for i, p in enumerate(precisions):
            if p == 0:
                smoothed_scores.append(1e-9)
            else:
                smoothed_scores.append(p)

        score_sum = sum(w * math.log(s) for w, s in zip(weights, smoothed_scores))
        bleu = brevity_penalty * math.exp(score_sum)
        return bleu


# =====================================================================
# 7. TRAINING & OPTIMIZATION LOOP ENGINE
# =====================================================================

class ModelTrainer:
    """
    A robust trainer featuring warmups, gradient clipping,
    early stopping, dynamic scheduling, and modular tracking.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 vocab: Vocabulary, device: torch.device, lr: float = 1e-3, 
                 checkpoint_dir: str = "./checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Cross Entropy loss with padding index masked out
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

    def train_epoch(self, epoch: int, teacher_forcing_ratio: float) -> float:
        """Trains the model for one full epoch across training loader datasets."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # RNN outputs: [Batch, Tgt_Len, VocabSize]
            # Transformer outputs: [Batch, Tgt_Len, VocabSize]
            vocab_dim = output.size(-1)
            
            # Reshape output/target matrices for multi-class classification
            # We omit index 0 matching <SOS> tokens from loss computation
            loss = self.criterion(output[:, 1:, :].reshape(-1, vocab_dim), tgt[:, 1:].reshape(-1))
            
            loss.backward()
            
            # Protect gradients against exploding thresholds
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.train_loader)

    def evaluate(self) -> float:
        """Validates model outputs across validation sets."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                # Evaluate model with teacher forcing deactivated (0.0)
                output = self.model(src, tgt, teacher_forcing_ratio=0.0)
                vocab_dim = output.size(-1)
                
                loss = self.criterion(output[:, 1:, :].reshape(-1, vocab_dim), tgt[:, 1:].reshape(-1))
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

    def fit(self, num_epochs: int = 10, initial_tf_ratio: float = 0.8, early_stopping_patience: int = 5) -> Dict[str, List[float]]:
        """Executes full optimization cycles, logging metrics to terminal."""
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        no_improvement_epochs = 0
        
        print(f"Beginning optimization on device: {self.device}")
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Decaying TF ratio over progressive epochs
            tf_ratio = max(0.1, initial_tf_ratio - (epoch - 1) * (initial_tf_ratio / num_epochs))
            
            train_loss = self.train_epoch(epoch, tf_ratio)
            val_loss = self.evaluate()
            
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            elapsed = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f} | TF: {tf_ratio:.2f} | Time: {elapsed:.2f}s")
            
            # Save checkpoints on improvements
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_reconstructor.pt"))
                print(" --> Improved model saved!")
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= early_stopping_patience:
                    print(f"Early stopping triggered after {early_stopping_patience} stagnant validation epochs.")
                    break
                    
        return history


# =====================================================================
# 8. HIGH-LEVEL API WRAPPER
# =====================================================================

class FrenchSentenceReconstructor:
    """
    User-facing high-level API orchestrating tokenizers, vocabularies, 
    neural model configurations, inference decoders, and metrics.
    """
    def __init__(self, architecture: str = "transformer", embed_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 4, 
                 device: Optional[str] = None):
        
        # Configure target compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.architecture = architecture.lower()
        self.tokenizer = FrenchSentenceTokenizer(lowercase=True, split_french_contractions=True)
        self.vocab = Vocabulary()
        self.corruptor = SentenceCorruptor(self.vocab)
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.model: Optional[nn.Module] = None
        self.beam_decoder: Optional[BeamSearchDecoder] = None

    def build_vocabulary(self, sentence_corpus: List[str], min_freq: int = 1):
        """Constructs mappings using training datasets."""
        tokenized_sentences = [self.tokenizer.tokenize(s) for s in sentence_corpus]
        self.vocab = Vocabulary(min_freq=min_freq)
        self.vocab.build_vocab(tokenized_sentences)
        self.corruptor = SentenceCorruptor(self.vocab)
        
        self._init_architecture()

    def _init_architecture(self):
        """Builds underlying neural model structures on active device configurations."""
        if self.architecture == "transformer":
            self.model = TransformerSeq2Seq(
                vocab_size=len(self.vocab),
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                ff_hidden_dim=self.hidden_dim,
                pad_idx=self.vocab.pad_idx,
                device=self.device
            ).to(self.device)
        elif self.architecture == "rnn":
            encoder = RNNAttentionEncoder(
                vocab_size=len(self.vocab),
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            ).to(self.device)
            decoder = RNNAttentionDecoder(
                vocab_size=len(self.vocab),
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            ).to(self.device)
            self.model = RNNAttentionSeq2Seq(encoder, decoder, self.device)
        else:
            raise ValueError(f"Unknown architecture option: '{self.architecture}'. Use 'transformer' or 'rnn'.")
            
        self.beam_decoder = BeamSearchDecoder(self.model, self.vocab, beam_size=4)

    def train(self, corpus: List[str], epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 1e-3, val_split: float = 0.15, 
              checkpoint_dir: str = "./checkpoints") -> Dict[str, List[float]]:
        """Fits our parameters to the input text datasets."""
        if not self.model:
            print("No vocabulary found. Instantiating mappings dynamically with min_freq=1...")
            self.build_vocabulary(corpus, min_freq=1)
            
        # Segment data splits
        random.seed(42)
        shuffled = list(corpus)
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1.0 - val_split))
        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]
        
        train_dataset = FrenchReconstructionDataset(train_data, self.tokenizer, self.vocab, self.corruptor)
        val_dataset = FrenchReconstructionDataset(val_data, self.tokenizer, self.vocab, self.corruptor)
        
        collate = DynamicPaddingCollate(self.vocab.pad_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        
        trainer = ModelTrainer(self.model, train_loader, val_loader, self.vocab, self.device, lr=learning_rate, checkpoint_dir=checkpoint_dir)
        history = trainer.fit(num_epochs=epochs)
        
        # Load best weights automatically
        best_path = os.path.join(checkpoint_dir, "best_reconstructor.pt")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            
        return history

    def reconstruct(self, sentence: str, method: str = "beam") -> str:
        """
        Cleans and reconstructs corrupted French sentences using selected decoding methods.
        Supported methods: 'greedy', 'beam'.
        """
        if self.model is None:
            raise RuntimeError("Model parameters are not initialized. Train or load models first.")
            
        self.model.eval()
        tokens = self.tokenizer.tokenize(sentence)
        src_indices = self.vocab.encode(tokens, add_sos=False, add_eos=True)
        src_tensor = torch.tensor([src_indices], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if method.lower() == "greedy":
                decoded_idx = self._greedy_decode(src_tensor)
            elif method.lower() == "beam":
                if self.architecture == "transformer":
                    decoded_idx = self.beam_decoder.decode_transformer(src_tensor)
                else:
                    decoded_idx = self.beam_decoder.decode_rnn(src_tensor)
            else:
                raise ValueError(f"Unsupported decoding method: '{method}'. Choose 'greedy' or 'beam'.")
                
        decoded_tokens = self.vocab.decode(decoded_idx)
        return self.tokenizer.untokenize(decoded_tokens)

    def _greedy_decode(self, src_tensor: torch.Tensor, max_len: int = 64) -> List[int]:
        """Runs fast greedy predictions across target sequence steps."""
        decoded_indices = [self.vocab.sos_idx]
        
        if self.architecture == "transformer":
            src_mask = self.model.make_src_mask(src_tensor)
            enc_outputs = self.model.encode(src_tensor, src_mask)
            
            for _ in range(max_len):
                tgt_tensor = torch.tensor([decoded_indices], dtype=torch.long, device=self.device)
                tgt_mask = self.model.make_tgt_mask(tgt_tensor)
                logits = self.model.decode(tgt_tensor, enc_outputs, tgt_mask, src_mask)
                
                next_word_idx = logits[0, -1, :].argmax(dim=-1).item()
                decoded_indices.append(next_word_idx)
                
                if next_word_idx == self.vocab.eos_idx:
                    break
        else:
            enc_outputs, hidden = self.model.encoder(src_tensor)
            dec_input = torch.tensor([[self.vocab.sos_idx]], dtype=torch.long, device=self.device)
            
            for _ in range(max_len):
                output, hidden = self.model.decoder(dec_input, hidden, enc_outputs)
                pred_idx = output.argmax(dim=-1).item()
                
                decoded_indices.append(pred_idx)
                if pred_idx == self.vocab.eos_idx:
                    break
                    
                dec_input = torch.tensor([[pred_idx]], dtype=torch.long, device=self.device)
                
        return decoded_indices

    def evaluate_on_test_pairs(self, test_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Measures performance statistics (BLEU, WER, Levenshtein Edit Distance)
        over a paired evaluation corpus.
        """
        bleu_scores = []
        wer_scores = []
        edit_distances = []
        
        for corrupted, ground_truth in test_pairs:
            reconstructed = self.reconstruct(corrupted, method="beam")
            
            tok_recon = self.tokenizer.tokenize(reconstructed)
            tok_truth = self.tokenizer.tokenize(ground_truth)
            
            bleu_scores.append(MetricSuite.sentence_bleu(tok_recon, tok_truth))
            wer_scores.append(MetricSuite.word_error_rate(tok_recon, tok_truth))
            edit_distances.append(MetricSuite.calculate_levenshtein(tok_recon, tok_truth))
            
        return {
            "avg_bleu": float(np.mean(bleu_scores)),
            "avg_wer": float(np.mean(wer_scores)),
            "avg_edit_distance": float(np.mean(edit_distances))
        }

    def save(self, directory: str):
        """Saves current state weights and tokenization maps to folder paths."""
        os.makedirs(directory, exist_ok=True)
        vocab_path = os.path.join(directory, "vocab.json")
        model_path = os.path.join(directory, "model.pt")
        config_path = os.path.join(directory, "config.json")
        
        self.vocab.save_to_file(vocab_path)
        torch.save(self.model.state_dict(), model_path)
        
        configs = {
            "architecture": self.architecture,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2)
            
        print(f"Configurations successfully serialized to folder: {directory}")

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "FrenchSentenceReconstructor":
        """Loads a model state checkpoint from directory."""
        vocab_path = os.path.join(directory, "vocab.json")
        model_path = os.path.join(directory, "model.pt")
        config_path = os.path.join(directory, "config.json")
        
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
            
        reconstructor = cls(
            architecture=configs["architecture"],
            embed_dim=configs["embed_dim"],
            hidden_dim=configs["hidden_dim"],
            num_layers=configs["num_layers"],
            num_heads=configs["num_heads"],
            device=device
        )
        
        reconstructor.vocab = Vocabulary.load_from_file(vocab_path)
        reconstructor._init_architecture()
        reconstructor.model.load_state_dict(torch.load(model_path, map_location=reconstructor.device))
        reconstructor.corruptor = SentenceCorruptor(reconstructor.vocab)
        
        print(f"Reconstructor successfully restored from: {directory}")
        return reconstructor


# =====================================================================
# 9. COMPREHENSIVE DIAGNOSTIC, TESTING, & BENCHMARKING ENGINE
# =====================================================================

def generate_synthetic_french_corpus() -> List[str]:
    """
    Generates a realistic, highly redundant set of French sentences with varying
    structures (interrogative, literary, descriptive) and accents to build
    a robust training corpus.
    """
    base_templates = [
        "L'intelligence artificielle transforme notre monde.",
        "Le chat dort paisiblement sur le canapé vert.",
        "Où se trouve la bibliothèque nationale de France ?",
        "Il faut manger pour vivre, et non vivre pour manger.",
        "Nous devons protéger l'environnement pour notre avenir.",
        "Le developpement de cette technologie prend du temps.",
        "Qu'avez-vous pensé de ce nouveau film français ?",
        "C'est une belle journée ensoleillée à Nice.",
        "Les enfants aiment jouer dehors pendant les vacances.",
        "La recherche scientifique apporte de nombreuses solutions.",
        "Comment vas-tu aujourd'hui mon cher ami ?",
        "L'ordinateur portable est un outil indispensable de nos jours.",
        "La musique adoucit les mœurs, dit-on souvent.",
        "Elle aime lire des romans historiques le soir.",
        "Le train de Paris arrivera à quai à seize heures.",
        "Je ne pense pas qu'il puisse venir ce soir.",
        "La gastronomie française est réputée dans le monde entier.",
        "Les étudiants apprennent la programmation en Python.",
        "Il y a beaucoup de vent aujourd'hui sur la côte.",
        "L'arbre devant la maison a perdu toutes ses feuilles.",
        "Avez-vous vu ce magnifique château médiéval ?",
        "La patience est la clé de la réussite en programmation.",
        "Nous irons visiter le musée du Louvre ce week-end.",
        "Le gâteau au chocolat que tu as fait est délicieux.",
        "L'apprentissage des langues étrangères ouvre l'esprit.",
        "Pourquoi le ciel est-il bleu pendant la journée ?",
        "Chaque problème a une solution, il suffit de la chercher.",
        "Le vélo est un excellent moyen de transport en ville.",
        "Ses parents habitent dans un petit village de campagne.",
        "Il a acheté de nouveaux livres pour étudier.",
        "L'art est le reflet de l'âme humaine.",
        "Le soleil se couche tard pendant l'été.",
        "Ma sœur apprend à jouer du piano depuis trois ans.",
        "Le café du coin sert d'excellents croissants chauds.",
        "Il ne faut jamais baisser les bras face aux difficultés.",
        "Nous apprécions beaucoup votre aide précieuse.",
        "Quel est ton plat français préféré ?",
        "La nature nous offre des paysages magnifiques en automne.",
        "Les réseaux de neurones imitent le cerveau humain.",
        "Ce projet demande une grande attention aux détails.",
        "On adore voyager en train à travers l'Europe.",
        "Le grand miroir du salon est cassé.",
        "Tu devrais écouter les conseils de ton médecin.",
        "La forêt est calme et mystérieuse la nuit.",
        "Le professeur explique les règles grammaticales.",
        "Elle a écrit une longue lettre à sa grand-mère.",
        "Où sont passées mes clés de voiture ?",
        "Les oiseaux chantent dès l'aube dans le jardin.",
        "Je préfère le thé noir au café au lait.",
        "Ce livre contient des recettes traditionnelles."
    ]
    
    # Let's perform deterministic syntax augmentation to grow the corpus to 800+ sentences
    # simulating realistic vocabulary distribution variations
    augmented_corpus = []
    pronouns = ["Je", "Tu", "Il", "Elle", "Nous", "Vous", "Ils", "Elles"]
    adjectives = ["grand", "petit", "magnifique", "nouveau", "vieux", "excellent"]
    verbs = ["pense à", "aime", "adore", "déteste", "cherche", "regarde", "prépare"]
    nouns = ["le projet", "la maison", "le train", "l'ordinateur", "la recette", "le livre"]

    for sentence in base_templates:
        augmented_corpus.append(sentence)
        
    for p in pronouns:
        for v in verbs:
            for n in nouns:
                augmented_corpus.append(f"{p} {v} {n} aujourd'hui.")
                
    for adj in adjectives:
        for item in ["ordinateur", "château", "miroir", "jardin", "village"]:
            augmented_corpus.append(f"C'est un {adj} {item} près d'ici.")
            
    # Add simple punctuation variation checks
    for base in base_templates[:20]:
        augmented_corpus.append(base.replace(".", " !"))
        augmented_corpus.append(base.replace(".", " ?"))
        
    return list(set(augmented_corpus)) # Deduplicate


if __name__ == "__main__":
    print("=" * 70)
    print("          FRENCH SENTENCE RECONSTRUCTION SYSTEM - TEST RUN")
    print("=" * 70)
    
    # Generate mock training samples
    french_corpus = generate_synthetic_french_corpus()
    print(f"Generated {len(french_corpus)} synthetic training sentences.")
    
    # Initialize our API wrapper utilizing the Transformer Seq2Seq setup
    print("\n[INFO] Initializing Reconstructor with a Custom Transformer architecture...")
    reconstructor = FrenchSentenceReconstructor(
        architecture="transformer",
        embed_dim=128,
        hidden_dim=256,
        num_layers=2, # Keep layers small for fast verification in tests
        num_heads=4,
        device="cpu"  # Force CPU to guarantee compatibility
    )
    
    # Build vocabulary mappings
    reconstructor.build_vocabulary(french_corpus, min_freq=1)
    
    # Select a small subsets of test validations
    test_cases_clean = [
        "L'intelligence artificielle transforme notre monde.",
        "Le chat dort paisiblement sur le canapé vert.",
        "Je préfère le thé noir au café au lait."
    ]
    
    # Display synthetic noise simulations on clean target patterns
    print("\n--- Spatially Simulating French Sentence Corruption Engine ---")
    for clean in test_cases_clean:
        toks = reconstructor.tokenizer.tokenize(clean)
        corrupted_toks = reconstructor.corruptor.corrupt_sentence(toks)
        corrupted_sentence = reconstructor.tokenizer.untokenize(corrupted_toks)
        print(f"Ground Truth : {clean}")
        print(f"Corrupted    : {corrupted_sentence}")
        print("-" * 50)
        
    # Fit the Model! (We run a brief set of epochs for demo confirmation)
    print("\n[INFO] Optimizing Neural Weights on Synthetic Corpus...")
    history = reconstructor.train(
        corpus=french_corpus,
        epochs=8,
        batch_size=32,
        learning_rate=0.003,
        val_split=0.1
    )
    
    # Construct a test evaluation set for diagnostic metrics validation
    test_eval_pairs = []
    for clean_sent in test_cases_clean:
        toks = reconstructor.tokenizer.tokenize(clean_sent)
        corrupted_toks = reconstructor.corruptor.corrupt_sentence(toks)
        corrupted_sent = reconstructor.tokenizer.untokenize(corrupted_toks)
        test_eval_pairs.append((corrupted_sent, clean_sent))
        
    print("\n--- Running Greedy vs. Beam Search Inference Decoders ---")
    for corrupted, ground_truth in test_eval_pairs:
        greedy_recon = reconstructor.reconstruct(corrupted, method="greedy")
        beam_recon = reconstructor.reconstruct(corrupted, method="beam")
        
        print(f"Corrupted Input : {corrupted}")
        print(f"Greedy Output   : {greedy_recon}")
        print(f"Beam Search     : {beam_recon}")
        print(f"Ground Truth    : {ground_truth}")
        print("-" * 50)
        
    # Run Metric Performance Benchmark Checks
    print("\n--- Metric Performance Evaluation Suite Diagnostics ---")
    metrics = reconstructor.evaluate_on_test_pairs(test_eval_pairs)
    print(f"Average BLEU Score : {metrics['avg_bleu']:.4f}")
    print(f"Average WER Score  : {metrics['avg_wer']:.4f}")
    print(f"Avg Edit Distance  : {metrics['avg_edit_distance']:.2f}")
    
    # Save the entire checkpoint setup
    temp_dir = "./fsr_checkpoint"
    print(f"\n[INFO] Saving serialized states to directory: {temp_dir}")
    reconstructor.save(temp_dir)
    
    # Re-load states from file path to verify persistence mechanics
    print("[INFO] Reloading checkpoint from disk to verify state continuity...")
    reloaded_reconstructor = FrenchSentenceReconstructor.load(temp_dir, device="cpu")
    
    # Final inference pass with reloaded model
    test_phrase = test_eval_pairs[0][0]
    expected = test_eval_pairs[0][1]
    reconstructed_text = reloaded_reconstructor.reconstruct(test_phrase, method="beam")
    print(f"\nLoaded Model Test:")
    print(f"Input:       {test_phrase}")
    print(f"Prediction:  {reconstructed_text}")
    print(f"Expected:    {expected}")
    
    print("\n" + "=" * 70)
    print("      FRENCH SENTENCE RECONSTRUCTION SYSTEM TEST RUN COMPLETE!")
    print("=" * 70)
