"""
Task 2: Text Engineering
- Cleaning, Tokenization, Vocabulary Creation, Sequence Padding, OOV Handling
AI Contract Intelligence System
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
PAD_TOKEN  = "<PAD>"   # padding token
UNK_TOKEN  = "<UNK>"   # out-of-vocabulary token
SOS_TOKEN  = "<SOS>"   # start of sequence
EOS_TOKEN  = "<EOS>"   # end of sequence

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

# Legal-domain stop words (keep domain-specific terms like 'liability', 'terminate')
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'it', 'its', 'this', 'that', 'these', 'those',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'or', 'and', 'but', 'if', 'then', 'so', 'yet', 'both', 'each',
    'either', 'all', 'any', 'no', 'not', 'such', 'than', 'too', 'very',
}


# ──────────────────────────────────────────────────────────────
# Step 1: Text Cleaning
# ──────────────────────────────────────────────────────────────
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean a contract text string.
    Steps:
        1. Lowercase
        2. Remove special characters (keep hyphens inside words)
        3. Normalize whitespace
        4. Optionally remove stop words
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)   # keep letters, digits, hyphens
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS]
        text = ' '.join(tokens)

    return text


# ──────────────────────────────────────────────────────────────
# Step 2: Tokenization
# ──────────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer with hyphen splitting.
    For legal text, preserves compound terms like 'non-compete'.
    """
    text = clean_text(text)
    tokens = text.split()
    return tokens


# ──────────────────────────────────────────────────────────────
# Step 3: Vocabulary Creation
# ──────────────────────────────────────────────────────────────
class Vocabulary:
    """
    Vocabulary class that maps tokens to integer IDs.

    Why vocabulary size matters:
    - Too small → many OOV tokens → information loss
    - Too large → sparse embeddings → memory and compute overhead
    - Ideal: cover ~95%+ of training corpus tokens
    """

    def __init__(self, max_vocab_size: int = 5000, min_freq: int = 1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.freq: Counter = Counter()
        self._built = False

        # Reserve indices for special tokens
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token2idx[tok] = i
            self.idx2token[i]   = tok

    @property
    def pad_idx(self):
        return self.token2idx[PAD_TOKEN]

    @property
    def unk_idx(self):
        return self.token2idx[UNK_TOKEN]

    @property
    def size(self):
        return len(self.token2idx)

    def build(self, corpus: List[str]) -> "Vocabulary":
        """
        Build vocabulary from a list of raw text strings.
        Only tokens above min_freq are included.
        Vocabulary is capped at max_vocab_size (most frequent first).
        """
        for text in corpus:
            self.freq.update(tokenize(text))

        # Filter by min_freq, sort by frequency (descending), cap at max_vocab_size
        eligible = [(tok, cnt) for tok, cnt in self.freq.items()
                    if cnt >= self.min_freq]
        eligible.sort(key=lambda x: x[1], reverse=True)
        eligible = eligible[:self.max_vocab_size - len(SPECIAL_TOKENS)]

        for tok, _ in eligible:
            if tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok

        self._built = True
        return self

    def encode(self, text: str) -> List[int]:
        """Convert text to list of integer IDs. Unknown tokens → UNK index."""
        return [self.token2idx.get(tok, self.unk_idx) for tok in tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        """Convert list of integer IDs back to text."""
        return ' '.join(self.idx2token.get(i, UNK_TOKEN) for i in ids
                        if i != self.pad_idx)

    def oov_rate(self, corpus: List[str]) -> float:
        """Compute OOV rate on a given corpus (0–1)."""
        total, oov = 0, 0
        for text in corpus:
            for tok in tokenize(text):
                total += 1
                if tok not in self.token2idx:
                    oov += 1
        return oov / total if total > 0 else 0.0

    def coverage(self) -> Dict:
        """Return vocabulary coverage statistics."""
        sorted_freq = sorted(self.freq.values(), reverse=True)
        total = sum(sorted_freq)
        vocab_total = sum(
            cnt for tok, cnt in self.freq.items() if tok in self.token2idx
        )
        return {
            "vocab_size": self.size,
            "total_unique_tokens": len(self.freq),
            "corpus_coverage_pct": round(vocab_total / total * 100, 2) if total else 0,
            "oov_types": len(self.freq) - (self.size - len(SPECIAL_TOKENS)),
        }


# ──────────────────────────────────────────────────────────────
# Step 4: Sequence Padding
# ──────────────────────────────────────────────────────────────
def pad_sequences(
    sequences: List[List[int]],
    max_len: int = None,
    padding: str = 'post',
    truncating: str = 'post',
    pad_value: int = 0
) -> np.ndarray:
    """
    Pad / truncate sequences to a fixed length.

    Why padding is required:
    - Neural networks (especially batch training) require fixed-size inputs.
    - Contracts have variable lengths; padding normalizes them.
    - <PAD> tokens carry no semantic meaning — attention masks exclude them.

    Args:
        sequences  : list of integer-encoded sequences
        max_len    : target length (defaults to longest in batch)
        padding    : 'post' (append) or 'pre' (prepend) padding
        truncating : 'post' (drop end) or 'pre' (drop start) for long seqs
        pad_value  : integer ID of PAD token (default 0)

    Returns:
        np.ndarray of shape (n_sequences, max_len)
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    result = np.full((len(sequences), max_len), pad_value, dtype=np.int64)

    for i, seq in enumerate(sequences):
        # Truncate if needed
        if len(seq) > max_len:
            seq = seq[:max_len] if truncating == 'post' else seq[-max_len:]

        if padding == 'post':
            result[i, :len(seq)] = seq
        else:  # pre-padding
            result[i, max_len - len(seq):] = seq

    return result


# ──────────────────────────────────────────────────────────────
# Step 5: OOV Handling strategy
# ──────────────────────────────────────────────────────────────
def handle_oov_strategy(
    text: str,
    vocab: Vocabulary,
    strategy: str = 'unk'
) -> List[int]:
    """
    OOV Handling strategies:
      'unk'      → replace unknown tokens with <UNK> id (standard approach)
      'subword'  → split unknown tokens into known sub-parts
      'skip'     → drop unknown tokens entirely

    In practice 'unk' is standard; subword tokenizers (BPE) are used in BERT.
    """
    tokens = tokenize(text)
    ids = []

    for tok in tokens:
        if tok in vocab.token2idx:
            ids.append(vocab.token2idx[tok])
        elif strategy == 'unk':
            ids.append(vocab.unk_idx)
        elif strategy == 'subword':
            # Try splitting on hyphens first, then character trigrams
            found = False
            if '-' in tok:
                parts = tok.split('-')
                for p in parts:
                    ids.append(vocab.token2idx.get(p, vocab.unk_idx))
                found = True
            if not found:
                ids.append(vocab.unk_idx)
        elif strategy == 'skip':
            pass  # simply omit

    return ids


# ──────────────────────────────────────────────────────────────
# Pipeline: process a full DataFrame
# ──────────────────────────────────────────────────────────────
def build_pipeline(df: pd.DataFrame, max_len: int = 64, max_vocab: int = 5000):
    """
    Full text engineering pipeline:
    1. Clean
    2. Tokenize
    3. Build vocabulary
    4. Encode + pad
    5. Report OOV stats
    """
    corpus = df["premise"].tolist() + df["hypothesis"].tolist()

    # Build vocab on full corpus
    vocab = Vocabulary(max_vocab_size=max_vocab, min_freq=1)
    vocab.build(corpus)

    # Encode premises and hypotheses
    premise_seqs  = [vocab.encode(t) for t in df["premise"].tolist()]
    hyp_seqs      = [vocab.encode(t) for t in df["hypothesis"].tolist()]

    # Pad to fixed length
    premise_padded = pad_sequences(premise_seqs,  max_len=max_len)
    hyp_padded     = pad_sequences(hyp_seqs,      max_len=max_len)

    # Attention masks (1 = real token, 0 = padding)
    premise_masks  = (premise_padded != vocab.pad_idx).astype(np.int32)
    hyp_masks      = (hyp_padded    != vocab.pad_idx).astype(np.int32)

    coverage = vocab.coverage()
    oov_rate = vocab.oov_rate(corpus)

    report = {
        "vocabulary_size"   : vocab.size,
        "unique_token_types": coverage["total_unique_tokens"],
        "corpus_coverage_%" : coverage["corpus_coverage_pct"],
        "oov_types"         : coverage["oov_types"],
        "oov_rate_%"        : round(oov_rate * 100, 2),
        "max_sequence_len"  : max_len,
        "premise_shape"     : premise_padded.shape,
        "hypothesis_shape"  : hyp_padded.shape,
    }

    return {
        "vocab"           : vocab,
        "premise_padded"  : premise_padded,
        "hyp_padded"      : hyp_padded,
        "premise_masks"   : premise_masks,
        "hyp_masks"       : hyp_masks,
        "report"          : report,
    }


if __name__ == "__main__":
    from task1_eda import build_dataset
    df = build_dataset(200)

    result = build_pipeline(df)
    print("\n=== Text Engineering Report ===")
    for k, v in result["report"].items():
        print(f"  {k:30s}: {v}")

    # Demo encode / decode
    vocab = result["vocab"]
    sample = df["premise"].iloc[0]
    encoded = vocab.encode(sample)
    decoded = vocab.decode(encoded)
    print(f"\nOriginal : {sample}")
    print(f"Encoded  : {encoded[:10]}...")
    print(f"Decoded  : {decoded}")
