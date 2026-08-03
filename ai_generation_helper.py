import re
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F

class SmartSampler:
    """
    Handles advanced neural logits sampling processes including Temperature scaling,
    Top-K truncation, and Top-P (Nucleus) cumulative filtering to eliminate gibberish.
    """
    @staticmethod
    def sample_logits(logits: torch.Tensor, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9) -> int:
        # Apply temperature scaling
        logits = logits / max(temperature, 1e-5)
        
        # Apply Top-K filtering
        if top_k > 0:
            v, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., -1, None]] = float('-inf')
            
        # Apply Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
            
        # Compute normalized probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Draw sample
        try:
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        except RuntimeError:
            # Fallback if all values became -inf
            next_token_id = torch.argmax(probs).item()
            
        return next_token_id


class FailsafeMarkovEngine:
    """
    A robust, backup text-generation model based on n-grams extracted dynamically
    from indexed files to guarantee syntactically correct text generation 
    even when PyTorch weights are totally untrained.
    """
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions: Dict[Tuple[str, ...], List[str]] = {}
        
    def fit_text(self, text: str):
        """Builds transition maps from text corpus, filtering out raw code lines."""
        clean_lines = []
        for line in text.split('\n'):
            line_strip = line.strip()
            # Skip python signatures, assignments, imports, and structural brackets to keep text natural
            if any(indicator in line_strip for indicator in ["def ", "class ", "import ", "self.", " = ", "torch.", "nn.", "plt.", "@"]):
                continue
            if line_strip.startswith("#") or line_strip.startswith("//") or line_strip.startswith("/*") or line_strip.startswith("*"):
                continue
            if not line_strip or len(line_strip) < 3:
                continue
            clean_lines.append(line_strip)
            
        cleaned_text = " ".join(clean_lines)
        words = [w for w in re.findall(r"\w+|[^\w\s]", cleaned_text, re.UNICODE) if w.strip()]
        if len(words) <= self.order:
            return
            
        for i in range(len(words) - self.order):
            state = tuple(words[i : i + self.order])
            next_word = words[i + self.order]
            if state not in self.transitions:
                self.transitions[state] = []
            self.transitions[state].append(next_word)

    def generate(self, seed_words: List[str], max_words: int = 50) -> str:
        """Generates sequence using transitions with safe randomized backoff."""
        if not self.transitions:
            return ""
            
        # Try to find a valid state in the seed text
        state = None
        for i in range(len(seed_words) - self.order, -1, -1):
            possible_state = tuple(seed_words[i : i + self.order])
            if possible_state in self.transitions:
                state = possible_state
                break
                
        if state is None:
            state = random.choice(list(self.transitions.keys()))
            
        result = list(state)
        for _ in range(max_words):
            if state in self.transitions:
                next_word = random.choice(self.transitions[state])
                result.append(next_word)
                state = tuple(result[-self.order:])
                if next_word in [".", "!", "?"]:
                    break
            else:
                # Random recovery jump
                state = random.choice(list(self.transitions.keys()))
                result.extend(list(state))
                
        return " ".join(result)


class AIOutputOptimizer:
    """
    Cleans up, normalizes and reconstructs beautiful text responses,
    removing artifacts and handling French contraction rules.
    """
    _french_words = None

    @staticmethod
    def load_dictionary(filepath: str = "words_french.txt") -> Set[str]:
        """Loads a list of valid French words from a file to use as a spelling reference."""
        import os
        if AIOutputOptimizer._french_words is not None:
            return AIOutputOptimizer._french_words
        
        words = set()
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        w = line.strip().lower()
                        if w:
                            words.add(w)
                AIOutputOptimizer._french_words = words
            except Exception:
                pass
        return words

    @staticmethod
    def edit_dist(s1: str, s2: str) -> int:
        """Computes Levenshtein edit distance between two strings to find close word matches."""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    @staticmethod
    def snap_to_dictionary(word: str, dictionary: Set[str]) -> str:
        """Snaps a predicted word to its closest match in words_french.txt if valid."""
        if not dictionary:
            return word
        
        # Isolate surrounding punctuation
        match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", word)
        if not match:
            return word
        prefix, clean_word, suffix = match.groups()
        
        if not clean_word:
            return word
            
        lower_word = clean_word.lower()
        if lower_word in dictionary:
            return word
            
        best_match = clean_word
        best_dist = 999
        
        # Look for the nearest word with an edit distance limit of 2
        for dict_word in dictionary:
            if abs(len(dict_word) - len(lower_word)) > 2:
                continue
            dist = AIOutputOptimizer.edit_dist(lower_word, dict_word)
            if dist < best_dist:
                best_dist = dist
                best_match = dict_word
                
        if best_dist <= 2:
            # Reconstruct original casing
            if clean_word.isupper():
                snapped = best_match.upper()
            elif clean_word[0].isupper():
                snapped = best_match.capitalize()
            else:
                snapped = best_match
            return prefix + snapped + suffix
            
        return word

    @staticmethod
    def optimize_text(raw_text: str, contraction_handler=None) -> str:
        if not raw_text:
            return ""
            
        # 1. Clean BPE & WordPiece space markings
        cleaned = raw_text.replace("</w>", " ")
        cleaned = cleaned.replace("##", "")
        
        # 2. De-duplicate multiple spaces and clean punctuation gaps
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
        
        # 3. Strip structural programming symbols leaked from code segments
        cleaned = re.sub(r"[\[\]{}()<>_\-=+*/|\\#]", "", cleaned)
        cleaned = re.sub(r"Source Code.*?Lines\s+\d+-\d+", "", cleaned)
        cleaned = re.sub(r"Document Segment.*?File.*?:", "", cleaned)
        
        # 4. Clean up consecutive spaces again after regex scrubbing
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        # 5. Apply French Contraction Handler if available
        if contraction_handler is not None:
            try:
                tokens = cleaned.split(" ")
                cleaned = contraction_handler.merge_contractions(tokens)
            except Exception:
                pass
                
        # 5b. Validate against words_french.txt dictionary if populated
        french_dict = AIOutputOptimizer.load_dictionary()
        if french_dict:
            try:
                tokens = cleaned.split(" ")
                corrected_tokens = [AIOutputOptimizer.snap_to_dictionary(t, french_dict) for t in tokens]
                cleaned = " ".join(corrected_tokens)
            except Exception:
                pass
                
        # 6. Final sanitization (ensure capital letters at beginning of sentences)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        final_sentences = []
        for s in sentences:
            if s:
                final_sentences.append(s[0].upper() + s[1:])
        return " ".join(final_sentences)
