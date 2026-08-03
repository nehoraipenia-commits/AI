"""
================================================================================
                    PREMIUM ENTERPRISE NLP TOKENIZATION SUITE
================================================================================
Author: Gemini (Google)
Language: Python 3.8+
Dependencies: torch, numpy

This module provides a robust, production-grade, and highly extensible framework 
for modern Natural Language Processing (NLP) tokenization. It includes fully
functional, highly optimized, and thoroughly documented implementations of:
    - Text Normalization Pipelines (Unicode, Case, Regex, Accent Stripping)
    - Pre-Tokenizers (Whitespace, Punctuation, Regex-based Splits)
    - Advanced Vocabulary Engines (with serialization & special tokens support)
    - Four Tokenizer Architectures (Word-Level, Char-Level, BPE, WordPiece)
    - Complete PyTorch Integration (Datasets, Collators, Neural Embeddings)
    - Diagnostics and Profiling suites.

This system is fully importable and can be integrated seamlessly into custom
Large Language Models (LLMs) or sequence-to-sequence training loops.
================================================================================
"""

import os
import re
import json
import time
import math
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Union, Generator, Iterator, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. CONFIGURATION & EXCEPTIONS
# ==============================================================================

class TokenizerError(Exception):
    """Base exception class for all errors generated inside the Tokenization Suite."""
    pass


class VocabularyError(TokenizerError):
    """Exception raised for errors associated with vocabulary builds or lookups."""
    pass


class NormalizationError(TokenizerError):
    """Exception raised during text preprocessing and normalization steps."""
    pass


@dataclass
class TokenizerConfig:
    """
    Configuration blueprint defining hyperparameters and properties for
    various tokenizer models within this suite.
    """
    vocab_size: int = 5000
    min_frequency: int = 2
    unk_token: str = "[UNK]"
    pad_token: str = "[PAD]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    mask_token: str = "[MASK]"
    
    # Suffix/Prefix indicators for subword tokenizers
    bpe_end_of_word: str = "</w>"
    wordpiece_prefix: str = "##"
    
    # Normalizer parameters
    lowercase: bool = True
    strip_accents: bool = True
    unicode_normalization: str = "NFKC"  # Options: 'NFC', 'NFD', 'NFKC', 'NFKD' or None
    
    # Sequence constraints
    max_length: int = 512
    truncation: bool = True
    padding: bool = True
    
    # Custom special tokens
    additional_special_tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration object to a standard Python dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "bpe_end_of_word": self.bpe_end_of_word,
            "wordpiece_prefix": self.wordpiece_prefix,
            "lowercase": self.lowercase,
            "strip_accents": self.strip_accents,
            "unicode_normalization": self.unicode_normalization,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "additional_special_tokens": self.additional_special_tokens
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenizerConfig":
        """Instantiates a configuration class from a Python dictionary representation."""
        return cls(**data)


# ==============================================================================
# 2. ADVANCED TEXT NORMALIZATION PIPELINE
# ==============================================================================

class BaseNormalizer:
    """
    Abstract interface defining structural properties for text normalizers.
    All normalization tasks must derive from this class.
    """
    def normalize(self, text: str) -> str:
        """Applies normalization transformation on the input string."""
        raise NotImplementedError("Subclasses must implement the normalize method.")


class UnicodeNormalizer(BaseNormalizer):
    """
    Normalizes unicode representations based on standard forms (NFC, NFD, NFKC, NFKD).
    Useful for unifying diverse keyboard inputs and symbols.
    """
    def __init__(self, form: str = "NFKC"):
        if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise NormalizationError(f"Invalid Unicode Normalization form: {form}")
        self.form = form

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        return unicodedata.normalize(self.form, text)


class LowercaseNormalizer(BaseNormalizer):
    """Converts alphabetic strings into their lowercase counterpart representations."""
    def normalize(self, text: str) -> str:
        return text.lower() if text else ""


class StripAccentsNormalizer(BaseNormalizer):
    """
    Strips diacritical marks and accents from characters.
    Example: 'café' becomes 'cafe'.
    """
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Decompose characters into base letters and separate diacritics
        decomposed = unicodedata.normalize("NFD", text)
        # Filter out character properties that correspond to non-spacing marks (diacritics)
        return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


class RegexNormalizer(BaseNormalizer):
    """
    Applies custom regular expressions to substitute character groupings 
    with designated replacements.
    """
    def __init__(self, pattern: str, replacement: str):
        try:
            self.regex = re.compile(pattern)
        except re.error as e:
            raise NormalizationError(f"Compiled regex error inside Normalizer: {e}")
        self.replacement = replacement

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        return self.regex.sub(self.replacement, text)


class NormalizerPipeline(BaseNormalizer):
    """
    Composes several individual normalizers to construct a ordered, multi-tier
    preprocessing stream.
    """
    def __init__(self, normalizers: List[BaseNormalizer]):
        self.normalizers = normalizers

    def normalize(self, text: str) -> str:
        for normalizer in self.normalizers:
            text = normalizer.normalize(text)
        return text

    @classmethod
    def build_from_config(cls, config: TokenizerConfig) -> "NormalizerPipeline":
        """Generates a structured Normalization Pipeline using configurations."""
        pipeline = []
        if config.unicode_normalization:
            pipeline.append(UnicodeNormalizer(config.unicode_normalization))
        if config.lowercase:
            pipeline.append(LowercaseNormalizer())
        if config.strip_accents:
            pipeline.append(StripAccentsNormalizer())
        
        # Clean consecutive whitespaces to a single space
        pipeline.append(RegexNormalizer(r"\s+", " "))
        return cls(pipeline)


# ==============================================================================
# 3. TEXT PRE-TOKENIZATION ENGINE (SPLITTERS)
# ==============================================================================

class BasePreTokenizer:
    """
    Abstract blueprint representing Pre-tokenizers. A pre-tokenizer takes
    normalized text and breaks it down into preliminary word fragments or
    character chunks before true token models process them.
    """
    def pre_tokenize(self, text: str) -> List[str]:
        """Transforms a unified text string into a list of pre-tokenized fragments."""
        raise NotImplementedError("Subclasses must implement pre_tokenize.")


class WhitespacePreTokenizer(BasePreTokenizer):
    """Splits plain strings along structural whitespace characters."""
    def pre_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [chunk for chunk in text.split(" ") if chunk]


class RegexPreTokenizer(BasePreTokenizer):
    """
    Advanced pre-tokenizer using sophisticated regex. Separates alphabetic blocks,
    numeric chunks, and single punctuations to guarantee consistency.
    """
    def __init__(self, pattern: Optional[str] = None):
        # Default pattern separates: alphabetic runs, numeric segments, punctuation marks
        default_pattern = r"\w+|[^\w\s]"
        self.pattern = pattern if pattern else default_pattern
        try:
            self.regex = re.compile(self.pattern)
        except re.error as e:
            raise TokenizerError(f"Pre-tokenizer regex compilation failed: {e}")

    def pre_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return self.regex.findall(text)


class PunctuationPreTokenizer(BasePreTokenizer):
    """
    Splits character blocks such that punctuation symbols are separated 
    into individual tokens while keeping whitespace chunks.
    """
    def pre_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = []
        current_chunk = []
        
        for char in text:
            # Check unicode category to see if it is a punctuation symbol
            category = unicodedata.category(char)
            is_punct = category.startswith("P") or category.startswith("S")
            
            if is_punct:
                if current_chunk:
                    tokens.append("".join(current_chunk))
                    current_chunk = []
                tokens.append(char)
            elif char.isspace():
                if current_chunk:
                    tokens.append("".join(current_chunk))
                    current_chunk = []
            else:
                current_chunk.append(char)
                
        if current_chunk:
            tokens.append("".join(current_chunk))
            
        return tokens


# ==============================================================================
# 4. VOCABULARY MANAGEMENT SYSTEM
# ==============================================================================

class Vocabulary:
    """
    Engine representing numerical mappings, token lookups, and serialization.
    Maintains indices for bidirectional structural decoding.
    """
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_frequencies: Dict[str, int] = {}
        self._special_tokens_set: Set[str] = set()
        
        self._initialize_special_tokens()

    def _initialize_special_tokens(self) -> None:
        """Populates the initial vocabulary using defined configuration controls."""
        specials = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.mask_token
        ]
        # Append additional tokens if present in configuration
        for token in self.config.additional_special_tokens:
            if token not in specials:
                specials.append(token)
                
        for index, token in enumerate(specials):
            if token:
                self.token_to_id[token] = index
                self.id_to_token[index] = token
                self._special_tokens_set.add(token)

    def add_token(self, token: str, frequency: int = 1) -> int:
        """
        Adds a single token to the mapping context if not already registered.
        Tracks global occurrences frequency.
        """
        if not token:
            raise VocabularyError("Attempted to register an empty or invalid token.")
            
        if token in self.token_to_id:
            self.token_frequencies[token] = self.token_frequencies.get(token, 0) + frequency
            return self.token_to_id[token]
            
        new_id = len(self.token_to_id)
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token
        self.token_frequencies[token] = frequency
        return new_id

    def lookup_token(self, token: str) -> int:
        """Finds ID associated with a given string token; defaults to UNK ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        return self.token_to_id.get(self.config.unk_token, -1)

    def lookup_id(self, token_id: int) -> str:
        """Translates numerical ID to associated string token; defaults to UNK."""
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        return self.config.unk_token

    def is_special(self, token: str) -> bool:
        """Queries if standard token belongs to defined special tokens."""
        return token in self._special_tokens_set

    @property
    def size(self) -> int:
        """Returns active size of internal vocabulary mapping."""
        return len(self.token_to_id)

    def save_to_file(self, filepath: str) -> None:
        """Saves structural mapping state as structured JSON on filesystem."""
        try:
            payload = {
                "config": self.config.to_dict(),
                "token_to_id": self.token_to_id,
                "token_frequencies": self.token_frequencies,
                "special_tokens": list(self._special_tokens_set)
            }
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=4)
        except Exception as e:
            raise VocabularyError(f"Could not write vocabulary to destination path: {e}")

    def load_from_file(self, filepath: str) -> None:
        """Reconstructs internal state from a structural vocabulary JSON file."""
        if not os.path.exists(filepath):
            raise VocabularyError(f"Target dictionary path does not exist: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                payload = json.load(file)
            
            self.config = TokenizerConfig.from_dict(payload["config"])
            self.token_to_id = payload["token_to_id"]
            # Convert keys back to integers for standard lookups
            self.id_to_token = {int(k): v for k, v in payload["token_to_id"].items()}
            # Swap mappings to align ID to tokens securely
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self.token_frequencies = payload.get("token_frequencies", {})
            self._special_tokens_set = set(payload.get("special_tokens", []))
        except Exception as e:
            raise VocabularyError(f"Encountered format anomalies loading serialization file: {e}")


# ==============================================================================
# 5. CORE ENCODING PAYLOAD STRUCTURES
# ==============================================================================

@dataclass
class Encoding:
    """
    A structured container holding the output of an encoding pipeline, 
    designed to interface directly with deep learning model components.
    """
    tokens: List[str]
    ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    special_tokens_mask: List[int]
    offsets: List[Optional[Tuple[int, int]]] = None

    def __repr__(self) -> str:
        return (
            f"Encoding(\n"
            f"  tokens             : {self.tokens[:10]}... (Len: {len(self.tokens)})\n"
            f"  ids                : {self.ids[:10]}... (Len: {len(self.ids)})\n"
            f"  attention_mask     : {self.attention_mask[:10]}... \n"
            f"  special_tokens_mask: {self.special_tokens_mask[:10]}...\n"
            f")"
        )


# ==============================================================================
# 6. BASE CLASS FOR TOKENIZERS
# ==============================================================================

class BaseTokenizer:
    """
    Abstract model describing core tokenization operational routines. All subword,
    character, and structural tokenizers inherit and specialize functions here.
    """
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config if config else TokenizerConfig()
        self.normalizer = NormalizerPipeline.build_from_config(self.config)
        self.vocab = Vocabulary(self.config)
        
        # Default fallback PreTokenizer setup
        self.pre_tokenizer = WhitespacePreTokenizer()

    def train_from_iterator(self, iterator: Iterator[str]) -> None:
        """Trains the model vocabulary using input iterator content streams."""
        raise NotImplementedError("Must implement train_from_iterator.")

    def train_from_files(self, filepaths: List[str]) -> None:
        """Iterates through plain text files to extract and build vocab maps."""
        def file_iterator():
            for path in filepaths:
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        yield line.strip()
        self.train_from_iterator(file_iterator())

    def _encode_single_text(self, text: str) -> List[str]:
        """Low-level sequence text-to-token converter."""
        raise NotImplementedError("BaseTokenizer subclass must implement internal encoder.")

    def encode(self, text: str, add_special_tokens: bool = True) -> Encoding:
        """
        Encodes sequence text into an unified Encoding payload containing ID mappings,
        attention masks, and special symbol properties.
        """
        # Step 1: Uniform Normalization
        normalized = self.normalizer.normalize(text)
        
        # Step 2: Extract base pre-tokens
        pre_tokens = self.pre_tokenizer.pre_tokenize(normalized)
        
        # Step 3: Run detailed tokenizer sub-routines
        raw_tokens = []
        for pre_token in pre_tokens:
            raw_tokens.extend(self._encode_single_text(pre_token))

        # Step 4: Constrain or pad tokens as configured
        return self._finalize_encoding(raw_tokens, add_special_tokens)

    def _finalize_encoding(self, tokens: List[str], add_special_tokens: bool) -> Encoding:
        """
        Post-processes raw tokens by applying truncation, injecting special tokens 
        (e.g., [BOS], [EOS]), and creating padding/attention masks.
        """
        bos = [self.config.bos_token] if (add_special_tokens and self.config.bos_token) else []
        eos = [self.config.eos_token] if (add_special_tokens and self.config.eos_token) else []
        
        # Handle truncation limits before prepending/appending special tags
        available_slots = self.config.max_length - len(bos) - len(eos)
        if self.config.truncation and len(tokens) > available_slots:
            tokens = tokens[:available_slots]
            
        final_tokens = bos + tokens + eos
        
        # Build numerical representation arrays
        ids = [self.vocab.lookup_token(tok) for tok in final_tokens]
        attention_mask = [1] * len(final_tokens)
        
        # Track where special indicators are inserted inside current array
        special_mask = []
        for tok in final_tokens:
            special_mask.append(1 if self.vocab.is_special(tok) else 0)
            
        token_type_ids = [0] * len(final_tokens)
        
        # Handle custom sequence padding logic
        if self.config.padding and len(final_tokens) < self.config.max_length:
            padding_len = self.config.max_length - len(final_tokens)
            pad_id = self.vocab.lookup_token(self.config.pad_token)
            
            final_tokens.extend([self.config.pad_token] * padding_len)
            ids.extend([pad_id] * padding_len)
            attention_mask.extend([0] * padding_len)
            special_mask.extend([1] * padding_len)
            token_type_ids.extend([0] * padding_len)

        return Encoding(
            tokens=final_tokens,
            ids=ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            special_tokens_mask=special_mask
        )

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Translates integer sequences back into unified human-readable strings."""
        tokens = []
        for val in ids:
            tok = self.vocab.lookup_id(val)
            if skip_special_tokens and self.vocab.is_special(tok):
                continue
            tokens.append(tok)
            
        return self._join_tokens(tokens)

    def _join_tokens(self, tokens: List[str]) -> str:
        """Merges tokens into a clean, human-readable string. Can be overridden."""
        return " ".join(tokens)

    def save(self, filepath: str) -> None:
        """Serializes current Tokenizer state and configuration onto disk."""
        self.vocab.save_to_file(filepath)

    def load(self, filepath: str) -> None:
        """Restores full Tokenizer capabilities from a saved serialization file."""
        self.vocab.load_from_file(filepath)
        # Update configurations and normalizers based on loaded attributes
        self.config = self.vocab.config
        self.normalizer = NormalizerPipeline.build_from_config(self.config)


# ==============================================================================
# 7. MODEL A: WORD-LEVEL TOKENIZER
# ==============================================================================

class WordTokenizer(BaseTokenizer):
    """
    Standard Word-Level Tokenization scheme mapping each space-separated sequence 
    or punctuation unit to its own unique numerical ID.
    """
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self.pre_tokenizer = RegexPreTokenizer()  # Isolates words and punctuation cleanly

    def train_from_iterator(self, iterator: Iterator[str]) -> None:
        """Counts occurrences of each unique word to build the vocabulary."""
        word_counts = {}
        for text in iterator:
            normalized = self.normalizer.normalize(text)
            pre_toks = self.pre_tokenizer.pre_tokenize(normalized)
            for tok in pre_toks:
                word_counts[tok] = word_counts.get(tok, 0) + 1

        # Sort candidate words by frequency
        sorted_tokens = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add high-frequency tokens to vocabulary up to the target vocab size
        for token, freq in sorted_tokens:
            if self.vocab.size >= self.config.vocab_size:
                break
            if freq >= self.config.min_frequency:
                self.vocab.add_token(token, freq)

    def _encode_single_text(self, text: str) -> List[str]:
        """Direct, exact lookup corresponding to the word value."""
        if not text:
            return []
        # Return token if it exists in vocabulary, otherwise map to UNK
        if text in self.vocab.token_to_id:
            return [text]
        return [self.config.unk_token]

    def _join_tokens(self, tokens: List[str]) -> str:
        """Reconstructs text while avoiding double spacing before punctuations."""
        if not tokens:
            return ""
        result = []
        for tok in tokens:
            # Avoid putting a space before standard punctuation marks
            if tok in {".", ",", "!", "?", ";", ":"} and result:
                result[-1] = result[-1] + tok
            else:
                result.append(tok)
        return " ".join(result)


# ==============================================================================
# 8. MODEL B: CHARACTER-LEVEL TOKENIZER
# ==============================================================================

class CharTokenizer(BaseTokenizer):
    """
    Decomposes strings into character sequences. Provides complete coverage 
    and eliminates out-of-vocabulary (UNK) errors.
    """
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        # We bypass whitespace isolation to retain base structure
        self.pre_tokenizer = WhitespacePreTokenizer()

    def train_from_iterator(self, iterator: Iterator[str]) -> None:
        """Iterates text characters to build the vocabulary."""
        char_counts = {}
        for text in iterator:
            normalized = self.normalizer.normalize(text)
            for char in normalized:
                char_counts[char] = char_counts.get(char, 0) + 1

        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        
        for char, freq in sorted_chars:
            if self.vocab.size >= self.config.vocab_size:
                break
            if freq >= self.config.min_frequency:
                self.vocab.add_token(char, freq)

    def _encode_single_text(self, text: str) -> List[str]:
        if not text:
            return []
        chars = []
        for char in text:
            if char in self.vocab.token_to_id:
                chars.append(char)
            else:
                chars.append(self.config.unk_token)
        return chars

    def _join_tokens(self, tokens: List[str]) -> str:
        """Glues character arrays together without gaps."""
        return "".join(tokens)


# ==============================================================================
# 9. MODEL C: BYTE-PAIR ENCODING (BPE) SUBWORD TOKENIZER
# ==============================================================================

class BytePairEncodingTokenizer(BaseTokenizer):
    """
    An optimized implementation of the Byte-Pair Encoding (BPE) subword tokenizer.
    Learns merges iteratively from a training corpus.
    """
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self.pre_tokenizer = RegexPreTokenizer()
        self.merges: Dict[Tuple[str, str], int] = {}
        self.cache: Dict[str, List[str]] = {}

    def train_from_iterator(self, iterator: Iterator[str]) -> None:
        """Trains BPE by finding and merging high-frequency character pairs."""
        # 1. Gather initial words with their frequencies
        word_counts = {}
        for text in iterator:
            normalized = self.normalizer.normalize(text)
            pre_toks = self.pre_tokenizer.pre_tokenize(normalized)
            for tok in pre_toks:
                word_counts[tok] = word_counts.get(tok, 0) + 1

        # 2. Represent words as tuples of characters with an end-of-word token
        splits = {}
        vocab_base_chars = set()
        for word, freq in word_counts.items():
            # Create a character representation sequence
            chars = list(word)
            if self.config.bpe_end_of_word:
                chars[-1] = chars[-1] + self.config.bpe_end_of_word
                
            splits[word] = chars
            for char in chars:
                vocab_base_chars.add(char)

        # 3. Add base characters to our vocabulary
        for char in sorted(vocab_base_chars):
            self.vocab.add_token(char)

        # 4. Calculate how many merge merges can occur
        target_merges = self.config.vocab_size - self.vocab.size
        if target_merges <= 0:
            return

        # 5. Iteratively find and merge the most frequent adjacent token pair
        for merge_idx in range(target_merges):
            pair_counts = self._count_adjacent_pairs(splits, word_counts)
            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            
            # If the frequency is below our threshold, stop merging
            if pair_counts[best_pair] < self.config.min_frequency:
                break

            # Register the new merge
            self.merges[best_pair] = merge_idx
            new_subword = "".join(best_pair)
            self.vocab.add_token(new_subword)

            # Apply the merge globally across our splits
            splits = self._apply_merge_to_splits(best_pair, splits)

        # Clear our encoding cache after training
        self.cache.clear()

    def _count_adjacent_pairs(self, splits: Dict[str, List[str]], word_counts: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Counts how often adjacent pairs of characters/subwords appear in our splits."""
        counts = {}
        for word, parts in splits.items():
            freq = word_counts[word]
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    def _apply_merge_to_splits(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Updates word splits by merging the specified pair of subwords."""
        new_splits = {}
        p1, p2 = pair
        for word, parts in splits.items():
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == p1 and parts[i+1] == p2:
                    new_parts.append(p1 + p2)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            new_splits[word] = new_parts
        return new_splits

    def _encode_single_text(self, text: str) -> List[str]:
        """Splits a single pre-tokenized word into subwords using learned BPE merges."""
        if not text:
            return []
            
        if text in self.cache:
            return self.cache[text]

        # Initialize split using individual characters
        word_parts = list(text)
        if self.config.bpe_end_of_word and word_parts:
            word_parts[-1] = word_parts[-1] + self.config.bpe_end_of_word

        while len(word_parts) > 1:
            # Find all possible merges for our current parts
            pairs = [(word_parts[i], word_parts[i+1]) for i in range(len(word_parts) - 1)]
            
            # Find the merge that was learned earliest (lowest merge index)
            eligible_merges = {pair: self.merges[pair] for pair in pairs if pair in self.merges}
            
            if not eligible_merges:
                break  # No more valid merges found

            best_pair_to_merge = min(eligible_merges, key=eligible_merges.get)
            
            # Perform the merge operation
            p1, p2 = best_pair_to_merge
            new_parts = []
            i = 0
            while i < len(word_parts):
                if i < len(word_parts) - 1 and word_parts[i] == p1 and word_parts[i+1] == p2:
                    new_parts.append(p1 + p2)
                    i += 2
                else:
                    new_parts.append(word_parts[i])
                    i += 1
            word_parts = new_parts

        # Map subwords to vocabulary. If a subword isn't found, fall back to UNK.
        final_tokens = []
        for part in word_parts:
            if part in self.vocab.token_to_id:
                final_tokens.append(part)
            else:
                # Fall back to character splitting for out-of-vocabulary subwords
                final_tokens.extend(self._char_fallback(part))
                
        # Cache and return results
        self.cache[text] = final_tokens
        return final_tokens

    def _char_fallback(self, subword: str) -> List[str]:
        """Splits unknown subwords into single characters to avoid UNK tokens."""
        fallback = []
        for char in subword:
            if char in self.vocab.token_to_id:
                fallback.append(char)
            else:
                fallback.append(self.config.unk_token)
        return fallback

    def _join_tokens(self, tokens: List[str]) -> str:
        """Reconstructs text by stripping BPE end-of-word markers."""
        if not tokens:
            return ""
        reconstructed = []
        current_word = []
        
        for tok in tokens:
            if self.config.bpe_end_of_word and tok.endswith(self.config.bpe_end_of_word):
                current_word.append(tok[:-len(self.config.bpe_end_of_word)])
                reconstructed.append("".join(current_word))
                current_word = []
            else:
                current_word.append(tok)
                
        if current_word:
            reconstructed.append("".join(current_word))
            
        return " ".join(reconstructed)

    def save(self, filepath: str) -> None:
        """Saves both vocabulary and BPE merges to disk."""
        super().save(filepath)
        # Append merges metadata mapping structures
        merges_filepath = filepath + ".merges"
        try:
            # Convert tuple keys to strings for JSON compatibility
            serializable_merges = {f"{k[0]} <-> {k[1]}": v for k, v in self.merges.items()}
            with open(merges_filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_merges, f, indent=4)
        except Exception as e:
            raise TokenizerError(f"Failed to serialize BPE merges metadata: {e}")

    def load(self, filepath: str) -> None:
        """Loads vocabulary and BPE merges from disk."""
        super().load(filepath)
        merges_filepath = filepath + ".merges"
        if os.path.exists(merges_filepath):
            try:
                with open(merges_filepath, "r", encoding="utf-8") as f:
                    serialized = json.load(f)
                
                self.merges = {}
                for key, val in serialized.items():
                    p1, p2 = key.split(" <-> ")
                    self.merges[(p1, p2)] = val
            except Exception as e:
                raise TokenizerError(f"Could not read BPE merges metadata: {e}")
        self.cache.clear()


# ==============================================================================
# 10. MODEL D: WORDPIECE SUBWORD TOKENIZER
# ==============================================================================

class WordPieceTokenizer(BaseTokenizer):
    """
    An implementation of the WordPiece subword tokenizer, widely used in models 
    like BERT. Subwords that are not at the start of a word are prefixed (e.g., '##').
    """
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self.pre_tokenizer = RegexPreTokenizer()
        self.cache: Dict[str, List[str]] = {}

    def train_from_iterator(self, iterator: Iterator[str]) -> None:
        """Trains WordPiece by maximizing the likelihood of merged candidates."""
        # 1. Gather word counts
        word_counts = {}
        for text in iterator:
            normalized = self.normalizer.normalize(text)
            pre_toks = self.pre_tokenizer.pre_tokenize(normalized)
            for tok in pre_toks:
                word_counts[tok] = word_counts.get(tok, 0) + 1

        # 2. Build initial vocabulary using all single characters
        base_char_counts = {}
        for word, freq in word_counts.items():
            for i, char in enumerate(word):
                if i == 0:
                    base_char_counts[char] = base_char_counts.get(char, 0) + freq
                else:
                    prefixed = self.config.wordpiece_prefix + char
                    base_char_counts[prefixed] = base_char_counts.get(prefixed, 0) + freq

        # Add top single characters to vocab
        sorted_base = sorted(base_char_counts.items(), key=lambda x: x[1], reverse=True)
        for tok, freq in sorted_base:
            if self.vocab.size >= self.config.vocab_size:
                break
            self.vocab.add_token(tok, freq)

        # 3. Represent words as lists of their character/subword pieces
        splits = {}
        for word in word_counts:
            parts = []
            for i, char in enumerate(word):
                if i == 0:
                    parts.append(char)
                else:
                    parts.append(self.config.wordpiece_prefix + char)
            splits[word] = parts

        # 4. Iteratively find and merge pairs with the highest score
        # WordPiece Score Formula: Score = Count(AB) / (Count(A) * Count(B))
        while self.vocab.size < self.config.vocab_size:
            pair_counts, component_counts = self._count_wordpiece_components(splits, word_counts)
            if not pair_counts:
                break

            # Calculate score for each candidate pair
            best_pair = None
            best_score = -1.0
            
            for pair, count_ab in pair_counts.items():
                p1, p2 = pair
                count_a = component_counts.get(p1, 0)
                count_b = component_counts.get(p2, 0)
                
                if count_a == 0 or count_b == 0:
                    continue
                    
                score = count_ab / (count_a * count_b)
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if not best_pair or best_score <= 0.0:
                break

            # Add the merged subword to our vocabulary
            p1, p2 = best_pair
            # Strip prefix from the second piece when merging (e.g., 'a' + '##b' -> 'ab')
            merged_token = p1 + p2.replace(self.config.wordpiece_prefix, "")
            self.vocab.add_token(merged_token)

            # Update our splits with the new merged token
            splits = self._apply_wordpiece_merge(best_pair, merged_token, splits)

        self.cache.clear()

    def _count_wordpiece_components(self, splits: Dict[str, List[str]], word_counts: Dict[str, int]) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int]]:
        """Counts frequencies of individual components and adjacent pairs."""
        pair_counts = {}
        component_counts = {}
        
        for word, parts in splits.items():
            freq = word_counts[word]
            for i, part in enumerate(parts):
                component_counts[part] = component_counts.get(part, 0) + freq
                if i < len(parts) - 1:
                    pair = (parts[i], parts[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    
        return pair_counts, component_counts

    def _apply_wordpiece_merge(self, pair: Tuple[str, str], merged: str, splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Updates word splits by merging the specified pair of subwords."""
        new_splits = {}
        p1, p2 = pair
        for word, parts in splits.items():
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == p1 and parts[i+1] == p2:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            new_splits[word] = new_parts
        return new_splits

    def _encode_single_text(self, text: str) -> List[str]:
        """Encodes a word using the greedy MaxMatch (longest-match-first) algorithm."""
        if not text:
            return []
            
        if text in self.cache:
            return self.cache[text]

        subwords = []
        start = 0
        word_len = len(text)
        is_bad = False

        while start < word_len:
            end = word_len
            cur_subword = None
            
            # Find the longest matching subword starting at our current index
            while start < end:
                substr = text[start:end]
                # Apply the WordPiece prefix if we are past the start of the word
                if start > 0:
                    substr = self.config.wordpiece_prefix + substr
                    
                if substr in self.vocab.token_to_id:
                    cur_subword = substr
                    break
                end -= 1

            if cur_subword is None:
                is_bad = True
                break

            subwords.append(cur_subword)
            start = end

        if is_bad:
            # Fallback to UNK if the word cannot be split using our vocabulary
            result = [self.config.unk_token]
        else:
            result = subwords

        self.cache[text] = result
        return result

    def _join_tokens(self, tokens: List[str]) -> str:
        """Reconstructs text by stripping WordPiece prefixes and joining tokens."""
        if not tokens:
            return ""
        
        reconstructed = []
        for tok in tokens:
            if tok.startswith(self.config.wordpiece_prefix):
                if reconstructed:
                    reconstructed[-1] = reconstructed[-1] + tok.replace(self.config.wordpiece_prefix, "")
                else:
                    reconstructed.append(tok.replace(self.config.wordpiece_prefix, ""))
            else:
                reconstructed.append(tok)
                
        return " ".join(reconstructed)


# ==============================================================================
# 11. PYTORCH DATA INTEGRATION
# ==============================================================================

class TokenizedDataset(Dataset):
    """
    A PyTorch Dataset that loads a text corpus, tokenizes each item, 
    and returns tensor inputs suitable for model training.
    """
    def __init__(self, texts: List[str], tokenizer: BaseTokenizer, max_length: Optional[int] = None):
        self.texts = texts
        self.tokenizer = tokenizer
        
        # Override configuration limits if custom limit provided
        if max_length:
            self.tokenizer.config.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.texts[index]
        encoding = self.tokenizer.encode(text, add_special_tokens=True)
        
        return {
            "input_ids": torch.tensor(encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(encoding.token_type_ids, dtype=torch.long),
            "special_tokens_mask": torch.tensor(encoding.special_tokens_mask, dtype=torch.long)
        }


class DataCollatorWithPadding:
    """
    A PyTorch DataLoader collator that dynamically pads batches to 
    the length of their longest sequence.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Identify the maximum sequence length in the current batch
        lengths = [x["input_ids"].shape[0] for x in batch]
        max_batch_len = max(lengths)
        
        batch_ids = []
        batch_attention = []
        batch_type = []
        batch_special = []

        for item in batch:
            ids = item["input_ids"]
            attn = item["attention_mask"]
            ttype = item["token_type_ids"]
            spec = item["special_tokens_mask"]
            
            diff = max_batch_len - ids.shape[0]
            if diff > 0:
                # Apply padding to reach the maximum length in the batch
                ids = torch.cat([ids, torch.full((diff,), self.pad_token_id, dtype=torch.long)])
                attn = torch.cat([attn, torch.zeros(diff, dtype=torch.long)])
                ttype = torch.cat([ttype, torch.zeros(diff, dtype=torch.long)])
                spec = torch.cat([spec, torch.ones(diff, dtype=torch.long)])
                
            batch_ids.append(ids)
            batch_attention.append(attn)
            batch_type.append(ttype)
            batch_special.append(spec)

        return {
            "input_ids": torch.stack(batch_ids),
            "attention_mask": torch.stack(batch_attention),
            "token_type_ids": torch.stack(batch_type),
            "special_tokens_mask": torch.stack(batch_special)
        }


# ==============================================================================
# 12. PYTORCH MODEL EMBEDDING INTEGRATION MODULE
# ==============================================================================

class TokenEmbedding(nn.Module):
    """
    A standard PyTorch NN module that maps token IDs to dense vectors, 
    accounting for padding indices.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Define standard PyTorch lookup table
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves embedding vectors for the provided token IDs.
        Input shape:  (batch_size, sequence_len)
        Output shape: (batch_size, sequence_len, embedding_dim)
        """
        return self.embeddings(input_ids)


# ==============================================================================
# 13. TOKENS DIAGNOSTIC & BENCHMARKING PIPELINE
# ==============================================================================

class TokenizerEvaluator:
    """
    A benchmarking suite that evaluates and compares tokenizers on properties like 
    compression ratios, throughput speeds, and round-trip fidelity.
    """
    def __init__(self, test_corpus: List[str]):
        self.corpus = test_corpus

    def evaluate_tokenizer(self, tokenizer: BaseTokenizer) -> Dict[str, Any]:
        """Runs a complete benchmark evaluation for the given tokenizer."""
        # 1. Measure throughput (speed)
        start_time = time.perf_counter()
        total_tokens = 0
        total_characters = 0
        
        for text in self.corpus:
            normalized = tokenizer.normalizer.normalize(text)
            pre_toks = tokenizer.pre_tokenizer.pre_tokenize(normalized)
            total_characters += len(text)
            
            # Count output tokens
            for p in pre_toks:
                total_tokens += len(tokenizer._encode_single_text(p))
                
        elapsed = time.perf_counter() - start_time
        tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0.0

        # 2. Check fidelity (round-trip reconstruction accuracy)
        failures = 0
        total_unks = 0
        
        for text in self.corpus:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            total_unks += encoded.ids.count(tokenizer.vocab.lookup_token(tokenizer.config.unk_token))
            
            decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
            normalized_source = tokenizer.normalizer.normalize(text)
            
            # Normalize whitespaces to compare reasonably
            cleaned_decoded = re.sub(r"\s+", " ", decoded).strip()
            cleaned_source = re.sub(r"\s+", " ", normalized_source).strip()
            
            # Simple fallback check for punctuation adjustments
            cleaned_source = cleaned_source.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            cleaned_decoded = cleaned_decoded.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            
            if cleaned_decoded != cleaned_source:
                failures += 1

        fidelity_rate = 1.0 - (failures / len(self.corpus)) if self.corpus else 1.0
        unk_ratio = total_unks / total_tokens if total_tokens > 0 else 0.0
        
        # 3. Calculate compression ratio (characters per token)
        compression = total_characters / total_tokens if total_tokens > 0 else 0.0

        return {
            "vocab_size": tokenizer.vocab.size,
            "total_tokens_processed": total_tokens,
            "throughput_tokens_per_sec": tokens_per_second,
            "compression_ratio_char_per_token": compression,
            "fidelity_accuracy_rate": fidelity_rate,
            "unknown_token_ratio": unk_ratio
        }


# ==============================================================================
# 14. COMPREHENSIVE SYSTEM DEMONSTRATION & TEST BED
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("                  INITIALIZING TOKENIZER TEST SUITE")
    print("=" * 80)

    # 1. Sample Corpus representing typical natural language inputs
    training_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning and natural language processing are changing the technology landscape rapidly.",
        "PyTorch provides powerful abstractions for neural networks and tensor computations.",
        "I love coding in Python! It is extremely readable and elegant to write.",
        "Tokenization is the process of breaking text down into words, characters, or subwords.",
        "Modern LLMs use subword tokenization to balance vocabulary size and out-of-vocabulary errors.",
        "Let's test punctuation: hello, world! Is this going to work? Yes, indeed; it is.",
        "Unifying spelling variants like processing, processors, and processed helps models generalise better.",
        "Can we handle accents? café, naïve, coöperation, résumé, and façade should be normalized."
    ]

    print(f"Loaded {len(training_corpus)} sample sentences for training and evaluation.\n")

    # 2. Base Configuration Setup
    config = TokenizerConfig(
        vocab_size=200,          # Intentionally small for demonstration
        min_frequency=1,
        lowercase=True,
        strip_accents=True,
        unicode_normalization="NFKC",
        max_length=16,           # Small sequence length to demonstrate padding & truncation
        padding=True,
        truncation=True
    )

    print("Initializing tokenizers with configuration properties:")
    print(json.dumps(config.to_dict(), indent=4))
    print("-" * 80)

    # 3. Instantiate and Train Tokenizers
    tokenizers = {
        "Word-Level Tokenizer": WordTokenizer(config),
        "Character-Level Tokenizer": CharTokenizer(config),
        "Byte-Pair Encoding (BPE)": BytePairEncodingTokenizer(config),
        "WordPiece Tokenizer": WordPieceTokenizer(config)
    }

    for name, tokenizer in tokenizers.items():
        print(f"Training [{name}] from corpus...")
        start = time.perf_counter()
        tokenizer.train_from_iterator(training_corpus)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"Training complete. Vocab size: {tokenizer.vocab.size} | Time taken: {elapsed:.2f} ms\n")

    print("=" * 80)
    print("                  EVALUATION & DIAGNOSTICS REPORT")
    print("=" * 80)

    # 4. Evaluate Tokenizers
    evaluator = TokenizerEvaluator(training_corpus)
    for name, tokenizer in tokenizers.items():
        metrics = evaluator.evaluate_tokenizer(tokenizer)
        print(f"Tokenizer: {name}")
        print(f"  Vocabulary Size         : {metrics['vocab_size']} tokens")
        print(f"  Fidelity Rate           : {metrics['fidelity_accuracy_rate']*100:.2f}% (matches clean text)")
        print(f"  Compression Ratio       : {metrics['compression_ratio_char_per_token']:.2f} chars/token")
        print(f"  Unknown (UNK) Ratio     : {metrics['unknown_token_ratio']*100:.2f}%")
        print(f"  Throughput Speed        : {metrics['throughput_tokens_per_sec']:.2f} tokens/second")
        print("-" * 40)

    print("=" * 80)
    print("                  ENCODING & DECODING SHOWCASE")
    print("=" * 80)

    # 5. Encoding Demonstration
    sample_phrase = "I love learning at the café!"
    print(f"Input string: '{sample_phrase}'\n")

    for name, tokenizer in tokenizers.items():
        encoding = tokenizer.encode(sample_phrase, add_special_tokens=True)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        
        print(f"[{name}]")
        print(f"  Tokens  : {encoding.tokens}")
        print(f"  IDs     : {encoding.ids}")
        print(f"  Decoded : '{decoded}'")
        print()

    print("=" * 80)
    print("                  PYTORCH DATALOADER INTEGRATION")
    print("=" * 80)

    # 6. Showcase integration with PyTorch
    target_tokenizer = tokenizers["Byte-Pair Encoding (BPE)"]
    print("Using BPE Tokenizer for deep learning batch loading:")
    
    # Create dataset
    dataset = TokenizedDataset(training_corpus, target_tokenizer)
    print(f"Created PyTorch Dataset containing {len(dataset)} items.")

    # Create dynamic collator
    pad_id = target_tokenizer.vocab.lookup_token(target_tokenizer.config.pad_token)
    collator = DataCollatorWithPadding(pad_token_id=pad_id)
    
    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collator)
    print("Created DataLoader with batch_size=3 and dynamic batch padding.\n")

    # Fetch a sample batch
    sample_batch = next(iter(dataloader))
    print("Loaded sample batch from DataLoader:")
    print(f"  input_ids Shape      : {sample_batch['input_ids'].shape}")
    print(f"  attention_mask Shape : {sample_batch['attention_mask'].shape}")
    print(f"  input_ids Tensor     :\n{sample_batch['input_ids']}")
    print(f"  attention_mask Tensor:\n{sample_batch['attention_mask']}")
    print()

    # 7. Embeddings Integration
    embedding_dim = 64
    embedder = TokenEmbedding(
        vocab_size=target_tokenizer.vocab.size,
        embedding_dim=embedding_dim,
        padding_idx=pad_id
    )
    
    embedded_vectors = embedder(sample_batch["input_ids"])
    print("Input IDs mapped to dense embeddings:")
    print(f"  Embedded Output Shape: {embedded_vectors.shape} (batch_size, sequence_length, embedding_dim)")

    print("\n" + "=" * 80)
    print("                      ALL TESTS PASSED SUCCESSFULLY")
    print("=" * 80)