"""
Task 4: Self-Attention Layer Model
Architecture: Input → Embedding → MultiHeadAttention → Dense → Output
AI Contract Intelligence System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from task3_baseline_model import ContractDataset, train_one_epoch, evaluate


# ──────────────────────────────────────────────────────────────
# Scaled Dot-Product Attention (from scratch)
# ──────────────────────────────────────────────────────────────
class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    Scaling by √d_k prevents gradients from vanishing in high-dimensional spaces
    where dot products grow large, pushing softmax into saturation regions.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V : (batch, heads, seq_len, d_k)
            mask    : (batch, 1, 1, seq_len) — 1 for valid, 0 for pad
        Returns:
            context : (batch, heads, seq_len, d_k)
            weights : (batch, heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Mask padding positions with large negative value (→ ~0 after softmax)
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)
        return context, weights


# ──────────────────────────────────────────────────────────────
# Multi-Head Attention
# ──────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Split embedding into `num_heads` sub-spaces, attend in each,
    then concatenate and project back.

    Different heads learn to attend to different aspects:
    e.g., one head might focus on legal verbs, another on named entities.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.d_k        = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout)
        self.norm      = nn.LayerNorm(embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def split_heads(self, x):
        """(B, L, D) → (B, H, L, D/H)"""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        """
        Self-attention: Q = K = V = x (same sequence attends to itself).
        Args:
            x    : (B, L, D)
            mask : (B, L) boolean — True for real tokens
        Returns:
            out     : (B, L, D)
            weights : (B, H, L, L)
        """
        B, L, _ = x.shape
        residual = x

        Q = self.split_heads(self.W_q(x))   # (B, H, L, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        if mask is not None:
            # Reshape mask: (B, L) → (B, 1, 1, L)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        context, weights = self.attention(Q, K, V, attn_mask)

        # Merge heads: (B, H, L, d_k) → (B, L, D)
        context = context.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        out = self.W_o(context)

        # Add & Norm (residual connection stabilises training)
        out = self.norm(residual + self.dropout(out))
        return out, weights


# ──────────────────────────────────────────────────────────────
# Feed-Forward sub-layer (standard Transformer block component)
# ──────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


# ──────────────────────────────────────────────────────────────
# Attention-based Contract NLI Model
# ──────────────────────────────────────────────────────────────
class AttentionModel(nn.Module):
    """
    Architecture:
        Input (token IDs)
          ↓
        Embedding [vocab_size × embed_dim]
          ↓
        MultiHeadAttention (self-attention over each sequence)
          ↓
        FeedForward sublayer
          ↓
        CLS-token pooling (first position)
          ↓
        Concatenate [premise_cls, hyp_cls]
          ↓
        Dense → Output [num_classes]
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 ff_dim: int    = 256,
                 num_classes: int = 3,
                 pad_idx: int  = 0,
                 dropout: float = 0.2):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Transformer-style encoder block (single layer)
        self.attention    = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def encode_sequence(self, token_ids):
        """
        Embed + attend + pool → single vector per sequence.
        Returns attended representation and attention weights.
        """
        mask  = (token_ids != self.pad_idx)       # (B, L)
        emb   = self.embedding(token_ids)          # (B, L, D)
        attn_out, weights = self.attention(emb, mask)
        ff_out = self.feed_forward(attn_out)

        # Mean pool (mask out PAD)
        mask_f = mask.float().unsqueeze(-1)        # (B, L, 1)
        pooled = (ff_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        return pooled, weights

    def forward(self, premise_ids, hyp_ids):
        p_vec, p_weights = self.encode_sequence(premise_ids)
        h_vec, h_weights = self.encode_sequence(hyp_ids)
        combined = torch.cat([p_vec, h_vec], dim=-1)
        logits   = self.classifier(combined)
        return logits, p_weights, h_weights


# ──────────────────────────────────────────────────────────────
# Training wrapper
# ──────────────────────────────────────────────────────────────
def train_epoch_attn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for premise, hyp, labels in loader:
        premise, hyp, labels = premise.to(device), hyp.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(premise, hyp)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(-1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate_attn(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for premise, hyp, labels in loader:
        premise, hyp = premise.to(device), hyp.to(device)
        logits, _, _ = model(premise, hyp)
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_attention_model(pipeline_result, df,
                          num_epochs=10, batch_size=32, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab   = pipeline_result["vocab"]
    premise = pipeline_result["premise_padded"]
    hyp     = pipeline_result["hyp_padded"]

    # Support both string labels ('e','n','c') and integer labels
    label_series = df["label"]
    if label_series.dtype == object or isinstance(label_series.iloc[0], str):
        label_map = {'e': 0, 'n': 1, 'c': 2,
                     'entailment': 0, 'neutral': 1, 'contradiction': 2}
        labels = np.array([label_map[str(v)[0].lower()] for v in label_series], dtype=np.int64)
    else:
        labels = label_series.values.astype(np.int64)

    n = len(labels)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    train_ds = ContractDataset(premise[tr_idx], hyp[tr_idx], labels[tr_idx])
    test_ds  = ContractDataset(premise[te_idx], hyp[te_idx], labels[te_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    model     = AttentionModel(vocab_size=vocab.size, pad_idx=vocab.pad_idx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        loss, acc = train_epoch_attn(model, train_dl, optimizer, criterion, device)
        preds, true_labels = evaluate_attn(model, test_dl, device)
        val_acc = accuracy_score(true_labels, preds)
        history["train_loss"].append(round(loss, 4))
        history["train_acc"].append(round(acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        scheduler.step()

    preds, true_labels = evaluate_attn(model, test_dl, device)
    label_names = ['entailment', 'neutral', 'contradiction']
    report = classification_report(true_labels, preds,
                                   labels=[0, 1, 2],
                                   target_names=label_names,
                                   output_dict=True, zero_division=0)

    metrics = {
        "accuracy"          : round(accuracy_score(true_labels, preds), 4),
        "classification_rep": report,
        "history"           : history,
        "label_names"       : label_names,
    }
    return model, metrics


if __name__ == "__main__":
    from task1_eda import build_dataset
    from task2_text_engineering import build_pipeline

    df     = build_dataset(200)
    result = build_pipeline(df)
    model, metrics = train_attention_model(result, df, num_epochs=5)
    print(f"\n=== Attention Model Results ===")
    print(f"Accuracy : {metrics['accuracy']}")
    rep = metrics["classification_rep"]
    for label in metrics["label_names"]:
        if label in rep:
            r = rep[label]
            print(f"  {label:15s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
