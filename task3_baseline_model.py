"""
Task 3: Baseline Model
Architecture: Input → Embedding → Dense → Output
AI Contract Intelligence System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


# ──────────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────────
class ContractDataset(Dataset):
    """PyTorch Dataset for contract NLI pairs."""

    def __init__(self, premise_ids: np.ndarray, hyp_ids: np.ndarray,
                 labels: np.ndarray):
        self.premise = torch.tensor(premise_ids, dtype=torch.long)
        self.hyp     = torch.tensor(hyp_ids,     dtype=torch.long)
        self.labels  = torch.tensor(labels,      dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.premise[idx], self.hyp[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────
# Baseline Model: Input → Embedding → Dense → Output
# ──────────────────────────────────────────────────────────────
class BaselineModel(nn.Module):
    """
    Simple baseline: mean-pooled embeddings → MLP classifier.

    Architecture:
        Input (premise + hypothesis token IDs)
          ↓
        Embedding Layer  [vocab_size × embed_dim]
          ↓
        Mean Pooling     [batch × embed_dim]
          ↓
        Concatenate      [batch × 2*embed_dim]
          ↓
        Dense (ReLU)     [2*embed_dim → hidden_dim]
          ↓
        Dropout
          ↓
        Dense (Output)   [hidden_dim → num_classes]
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 pad_idx: int = 0,
                 dropout: float = 0.3):
        super().__init__()

        self.pad_idx = pad_idx

        # Embedding layer — maps each token ID to a dense vector
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx    # PAD embeddings are zero and not updated
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def mean_pool(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Average embeddings, ignoring PAD tokens."""
        emb = self.embedding(token_ids)                    # (B, L, D)
        mask = (token_ids != self.pad_idx).float()         # (B, L)
        mask = mask.unsqueeze(-1)                          # (B, L, 1)
        summed = (emb * mask).sum(dim=1)                   # (B, D)
        lengths = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
        return summed / lengths                            # (B, D)

    def forward(self, premise_ids: torch.Tensor,
                hyp_ids: torch.Tensor) -> torch.Tensor:
        premise_emb = self.mean_pool(premise_ids)          # (B, D)
        hyp_emb     = self.mean_pool(hyp_ids)              # (B, D)
        combined    = torch.cat([premise_emb, hyp_emb], dim=-1)  # (B, 2D)
        logits      = self.classifier(combined)            # (B, C)
        return logits


# ──────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for premise, hyp, labels in loader:
        premise, hyp, labels = premise.to(device), hyp.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(premise, hyp)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds  = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for premise, hyp, labels in loader:
        premise, hyp = premise.to(device), hyp.to(device)
        logits = model(premise, hyp)
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_baseline(pipeline_result: Dict,
                   df: pd.DataFrame,
                   num_epochs: int = 10,
                   batch_size: int = 32,
                   lr: float = 1e-3) -> Tuple[BaselineModel, Dict]:
    """
    Train the baseline model and return trained model + metrics dict.
    """
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

    # Simple train/test split (80/20)
    n = len(labels)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    train_ds = ContractDataset(premise[tr_idx], hyp[tr_idx], labels[tr_idx])
    test_ds  = ContractDataset(premise[te_idx], hyp[te_idx], labels[te_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    model     = BaselineModel(vocab_size=vocab.size, pad_idx=vocab.pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
        preds, true_labels = evaluate(model, test_dl, device)
        val_acc = accuracy_score(true_labels, preds)
        history["train_loss"].append(round(loss, 4))
        history["train_acc"].append(round(acc, 4))
        history["val_acc"].append(round(val_acc, 4))

    # Final evaluation
    preds, true_labels = evaluate(model, test_dl, device)
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
    import json
    from task1_eda import build_dataset
    from task2_text_engineering import build_pipeline

    df     = build_dataset(200)
    result = build_pipeline(df)
    model, metrics = train_baseline(result, df, num_epochs=5)

    print(f"\n=== Baseline Model Results ===")
    print(f"Accuracy : {metrics['accuracy']}")
    rep = metrics["classification_rep"]
    for label in metrics["label_names"]:
        if label in rep:
            r = rep[label]
            print(f"  {label:15s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
