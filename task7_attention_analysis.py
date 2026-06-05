"""
Task 7: Attention Analysis
- Display attention scores
- Highlight most important words
- Examples: payment, termination, confidential
AI Contract Intelligence System
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict, Optional
import re


# ──────────────────────────────────────────────────────────────
# Simple attention extractor (works with any model that has MHA)
# ──────────────────────────────────────────────────────────────
def get_attention_weights(
    model, vocab, premise: str, hypothesis: str,
    device: str = "cpu", max_len: int = 64
) -> Dict:
    """
    Run a single forward pass and extract attention weights.
    Returns dict with premise/hypothesis tokens and attention matrices.
    """
    from task2_text_engineering import tokenize, pad_sequences

    model.eval()
    model = model.to(device)

    p_tokens = tokenize(premise)
    h_tokens = tokenize(hypothesis)

    p_ids = [vocab.token2idx.get(t, vocab.unk_idx) for t in p_tokens]
    h_ids = [vocab.token2idx.get(t, vocab.unk_idx) for t in h_tokens]

    p_padded = pad_sequences([p_ids], max_len=max_len)
    h_padded = pad_sequences([h_ids], max_len=max_len)

    p_tensor = torch.tensor(p_padded, dtype=torch.long).to(device)
    h_tensor = torch.tensor(h_padded, dtype=torch.long).to(device)

    with torch.no_grad():
        logits, p_weights, h_weights = model(p_tensor, h_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    # Trim weights to actual token length (remove padding positions)
    p_len = len(p_tokens)
    h_len = len(h_tokens)

    # p_weights shape: (1, num_heads, seq_len, seq_len)
    p_attn = p_weights[0].cpu().numpy()  # (H, L, L)
    h_attn = h_weights[0].cpu().numpy()  # (H, L, L)

    # Mean across heads
    p_attn_mean = p_attn.mean(axis=0)[:p_len, :p_len]   # (p_len, p_len)
    h_attn_mean = h_attn.mean(axis=0)[:h_len, :h_len]   # (h_len, h_len)

    # Token importance = column-sum of attention received
    p_importance = p_attn_mean.sum(axis=0)
    h_importance = h_attn_mean.sum(axis=0)

    # Normalize
    p_importance = p_importance / (p_importance.sum() + 1e-9)
    h_importance = h_importance / (h_importance.sum() + 1e-9)

    label_names = ['entailment', 'neutral', 'contradiction']
    pred_label  = label_names[probs.argmax()]

    return {
        "premise_tokens"     : p_tokens,
        "hypothesis_tokens"  : h_tokens,
        "premise_attn"       : p_attn_mean,
        "hypothesis_attn"    : h_attn_mean,
        "premise_importance" : p_importance,
        "hypothesis_importance": h_importance,
        "probabilities"      : dict(zip(label_names, probs.tolist())),
        "prediction"         : pred_label,
        "all_heads_premise"  : p_attn[:, :p_len, :p_len],
        "all_heads_hyp"      : h_attn[:, :h_len, :h_len],
    }


# ──────────────────────────────────────────────────────────────
# Attention Heatmap
# ──────────────────────────────────────────────────────────────
def plot_attention_heatmap(attn_matrix: np.ndarray,
                           row_labels: List[str],
                           col_labels: List[str],
                           title: str = "Attention Weights",
                           save_path: str = None,
                           figsize: tuple = (10, 8)):
    """
    Plot a heatmap of attention scores.
    Row i, column j = how much token i attends to token j.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto',
                   vmin=0, vmax=attn_matrix.max())

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells
    for i in range(attn_matrix.shape[0]):
        for j in range(attn_matrix.shape[1]):
            val = attn_matrix[i, j]
            color = 'white' if val > 0.5 * attn_matrix.max() else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Attention weight')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_token_importance(tokens: List[str], importance: np.ndarray,
                          title: str = "Token Importance",
                          save_path: str = None,
                          highlight_words: List[str] = None):
    """
    Bar chart showing how important each token is (by attention received).
    Key legal terms (payment, termination, confidential) are highlighted.
    """
    if highlight_words is None:
        highlight_words = ['payment', 'termination', 'confidential',
                           'terminate', 'liability', 'compete', 'disclose',
                           'indemnify', 'breach', 'penalty']

    colors = []
    for tok in tokens:
        if any(hw in tok.lower() for hw in highlight_words):
            colors.append('#e74c3c')   # red = key legal term
        else:
            colors.append('#3498db')   # blue = regular term

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.8), 5))
    bars = ax.bar(range(len(tokens)), importance, color=colors, edgecolor='white')

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Attention Score (normalised)")
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Value labels on bars
    for bar, val in zip(bars, importance):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Key Legal Term'),
                       Patch(facecolor='#3498db', label='Other Token')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_all_heads(all_heads: np.ndarray, tokens: List[str],
                   title: str = "All Attention Heads",
                   save_path: str = None):
    """Show attention pattern for every head in a grid."""
    n_heads = all_heads.shape[0]
    cols    = min(4, n_heads)
    rows    = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).flatten()

    for h in range(n_heads):
        ax  = axes[h]
        mat = all_heads[h]
        ax.imshow(mat, cmap='Blues', aspect='auto',
                  vmin=0, vmax=mat.max() + 1e-9)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f'Head {h+1}', fontsize=9, fontweight='bold')

    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────────────
# Demo — rule-based attention for visualization when model unavailable
# ──────────────────────────────────────────────────────────────
KEY_LEGAL_TERMS = {
    'payment': 0.95, 'termination': 0.93, 'confidential': 0.91,
    'liability': 0.88, 'compete': 0.85, 'terminate': 0.87,
    'disclose': 0.83, 'indemnify': 0.82, 'breach': 0.80,
    'penalty': 0.78, 'notice': 0.72, 'days': 0.68,
    'party': 0.60, 'agreement': 0.58, 'shall': 0.55,
}


def compute_rule_based_importance(tokens: List[str]) -> np.ndarray:
    """
    Heuristic attention: assigns high scores to known legal keywords.
    Used for demo / when model is not yet trained.
    """
    scores = np.array([
        KEY_LEGAL_TERMS.get(t.lower(), 0.1 + np.random.uniform(0, 0.2))
        for t in tokens
    ])
    scores = scores / scores.sum()
    return scores


def demo_attention_analysis(save_dir: str = "attention_outputs"):
    """Run demo attention analysis on example legal clauses."""
    os.makedirs(save_dir, exist_ok=True)

    examples = [
        {
            "premise"   : "Payment shall be made within 30 days of invoice receipt.",
            "hypothesis": "The company must pay within one month.",
            "label"     : "entailment",
        },
        {
            "premise"   : "Termination requires 60 days written notice from either party.",
            "hypothesis": "The contract can be terminated without any notice.",
            "label"     : "contradiction",
        },
        {
            "premise"   : "All confidential information shall remain strictly private.",
            "hypothesis": "Sensitive data must be protected from disclosure.",
            "label"     : "entailment",
        },
    ]

    results = []
    for i, ex in enumerate(examples):
        p_tokens = ex["premise"].lower().replace('.', '').split()
        h_tokens = ex["hypothesis"].lower().replace('.', '').split()

        p_imp = compute_rule_based_importance(p_tokens)
        h_imp = compute_rule_based_importance(h_tokens)

        # Build fake attention matrix (outer product weighted by importance)
        p_attn = np.outer(p_imp, p_imp)
        p_attn = p_attn / p_attn.sum(axis=1, keepdims=True)

        # Plot
        plot_attention_heatmap(
            p_attn, p_tokens, p_tokens,
            title=f"Attention Map — Premise\n\"{ex['premise'][:60]}...\"",
            save_path=os.path.join(save_dir, f"attn_heatmap_{i}.png")
        )
        plot_token_importance(
            p_tokens, p_imp,
            title=f"Token Importance — Premise ({ex['label']})",
            save_path=os.path.join(save_dir, f"token_importance_{i}.png")
        )
        plt.close('all')

        results.append({
            "premise"           : ex["premise"],
            "hypothesis"        : ex["hypothesis"],
            "label"             : ex["label"],
            "premise_tokens"    : p_tokens,
            "hypothesis_tokens" : h_tokens,
            "premise_importance": p_imp,
            "hyp_importance"    : h_imp,
        })

    return results


import os

if __name__ == "__main__":
    results = demo_attention_analysis()
    print("\n=== Attention Analysis Demo ===")
    for r in results:
        print(f"\n[{r['label'].upper()}]")
        print(f"  Premise    : {r['premise']}")
        print(f"  Hypothesis : {r['hypothesis']}")
        tokens_imp = sorted(
            zip(r["premise_tokens"], r["premise_importance"]),
            key=lambda x: x[1], reverse=True
        )[:5]
        print(f"  Top tokens : {[(t, round(s, 3)) for t, s in tokens_imp]}")
