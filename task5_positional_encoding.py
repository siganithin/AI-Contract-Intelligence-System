"""
Task 5: Positional Encoding — implemented from scratch
Task 6: Clause Understanding Analysis (same & different word order)
AI Contract Intelligence System
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


# ──────────────────────────────────────────────────────────────
# Positional Encoding (Vaswani et al. 2017)
# ──────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.

    Formulas:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Why:
    - Transformers have no inherent notion of order (unlike RNNs).
    - PE encodes each position as a unique pattern of sine/cosine waves.
    - Close positions → similar patterns; distant positions → different patterns.
    - Allows the model to learn positional relationships.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encoding matrix once
        pe = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-np.log(10000.0) / d_model)
        )                                                        # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)           # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)           # odd  dims → cos

        pe = pe.unsqueeze(0)                                    # (1, max_len, d_model)
        self.register_buffer('pe', pe)                          # not a parameter, but saved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, d_model) — token embeddings
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

    def get_matrix(self, seq_len: int = None, d_model: int = None) -> np.ndarray:
        """Return the PE matrix as numpy array for visualization."""
        pe = self.pe.squeeze(0).cpu().numpy()                  # (max_len, d_model)
        if seq_len:
            pe = pe[:seq_len]
        if d_model:
            pe = pe[:, :d_model]
        return pe


# ──────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────
def plot_positional_encoding(pe_matrix: np.ndarray, save_path: str = None,
                             positions_to_highlight: list = None):
    """
    Heatmap of positional encoding matrix.
    Each row = one position, each column = one embedding dimension.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Full heatmap
    im = axes[0].imshow(pe_matrix, aspect='auto', cmap='RdYlBu',
                        vmin=-1, vmax=1, interpolation='nearest')
    axes[0].set_title("Positional Encoding Matrix\n(rows=positions, cols=dimensions)",
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_ylabel("Position")
    plt.colorbar(im, ax=axes[0], label='PE value')

    # Highlight specific positions
    if positions_to_highlight:
        colors = plt.cm.tab10(np.linspace(0, 1, len(positions_to_highlight)))
        for pos, color in zip(positions_to_highlight, colors):
            if pos < pe_matrix.shape[0]:
                axes[1].plot(pe_matrix[pos, :],
                             label=f'Position {pos}',
                             color=color, linewidth=2)
        axes[1].set_title("PE Values Across Dimensions\nfor Selected Positions",
                          fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Embedding Dimension")
        axes[1].set_ylabel("PE Value")
        axes[1].legend(loc='upper right')
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[1].set_ylim(-1.1, 1.1)
    else:
        axes[1].plot(pe_matrix[:, 0], label='Dim 0 (sin)', color='steelblue')
        axes[1].plot(pe_matrix[:, 1], label='Dim 1 (cos)', color='coral')
        axes[1].plot(pe_matrix[:, 4], label='Dim 4', color='green')
        axes[1].set_title("PE Values Across Positions (Selected Dims)",
                          fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Position")
        axes[1].set_ylabel("PE Value")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_per_position_heatmaps(pe_matrix: np.ndarray, num_positions: int = 5,
                               save_path: str = None):
    """
    Individual heatmap for each of the first N positions.
    Visualises position 1, 2, 3, ... separately.
    """
    cols  = min(num_positions, 5)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))

    if cols == 1:
        axes = [axes]

    for i in range(cols):
        pos = i  # 0-indexed position
        row = pe_matrix[pos, :].reshape(1, -1)  # (1, d_model)
        axes[i].imshow(row, aspect='auto', cmap='RdYlBu', vmin=-1, vmax=1)
        axes[i].set_title(f'Position {pos + 1}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Dimension')
        axes[i].set_yticks([])
        # Annotate selected values
        d = row.shape[1]
        for j in range(min(8, d)):
            axes[i].text(j, 0, f'{row[0, j]:.2f}', ha='center', va='center',
                         fontsize=6, color='black')

    plt.suptitle('Positional Encoding — Per-Position Heatmaps',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────────────
# Task 6: Clause Understanding Analysis
# ──────────────────────────────────────────────────────────────
CLAUSE_A = "Payment shall be made within 30 days."
CLAUSE_B = "Within 30 days payment shall be made."


def analyse_word_order(sentence_a: str = CLAUSE_A,
                       sentence_b: str = CLAUSE_B,
                       d_model: int = 64,
                       save_path: str = None) -> dict:
    """
    Show how the same words at different positions produce different
    positional representations — demonstrating why order matters.
    """
    pe_encoder = PositionalEncoding(d_model=d_model, max_len=64, dropout=0.0)
    pe_encoder.eval()

    # Tokenise both sentences
    tokens_a = sentence_a.lower().replace('.', '').split()
    tokens_b = sentence_b.lower().replace('.', '').split()

    # Dummy zero embeddings — we only care about PE contribution
    emb_a = torch.zeros(1, len(tokens_a), d_model)
    emb_b = torch.zeros(1, len(tokens_b), d_model)

    with torch.no_grad():
        pe_a = pe_encoder(emb_a).squeeze(0).numpy()   # (len_a, d_model)
        pe_b = pe_encoder(emb_b).squeeze(0).numpy()   # (len_b, d_model)

    # Find common words and compare their PE vectors
    common_words = set(tokens_a) & set(tokens_b)
    word_analysis = {}
    for word in sorted(common_words):
        pos_in_a = tokens_a.index(word) if word in tokens_a else None
        pos_in_b = tokens_b.index(word) if word in tokens_b else None
        if pos_in_a is not None and pos_in_b is not None:
            vec_a = pe_a[pos_in_a]
            vec_b = pe_b[pos_in_b]
            cosine_sim = float(
                np.dot(vec_a, vec_b) /
                (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9)
            )
            word_analysis[word] = {
                "position_in_A": pos_in_a + 1,
                "position_in_B": pos_in_b + 1,
                "pe_diff_norm" : round(float(np.linalg.norm(vec_a - vec_b)), 4),
                "cosine_sim"   : round(cosine_sim, 4),
            }

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im_a = axes[0].imshow(pe_a, aspect='auto', cmap='RdYlBu', vmin=-1, vmax=1)
    axes[0].set_title(f'Contract A Positional Encodings\n"{sentence_a}"',
                      fontsize=11, fontweight='bold')
    axes[0].set_yticks(range(len(tokens_a)))
    axes[0].set_yticklabels(tokens_a, fontsize=9)
    axes[0].set_xlabel('Dimension')
    plt.colorbar(im_a, ax=axes[0])

    im_b = axes[1].imshow(pe_b, aspect='auto', cmap='RdYlBu', vmin=-1, vmax=1)
    axes[1].set_title(f'Contract B Positional Encodings\n"{sentence_b}"',
                      fontsize=11, fontweight='bold')
    axes[1].set_yticks(range(len(tokens_b)))
    axes[1].set_yticklabels(tokens_b, fontsize=9)
    axes[1].set_xlabel('Dimension')
    plt.colorbar(im_b, ax=axes[1])

    plt.suptitle('Same Words — Different Positions → Different PE Vectors',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return {
        "sentence_a"    : sentence_a,
        "sentence_b"    : sentence_b,
        "tokens_a"      : tokens_a,
        "tokens_b"      : tokens_b,
        "word_analysis" : word_analysis,
        "pe_a"          : pe_a,
        "pe_b"          : pe_b,
        "figure"        : fig,
        "explanation"   : (
            "Both sentences contain identical words, but their order differs. "
            "Because each word receives the PE vector of its position (not its identity), "
            "the word 'payment' at position 1 in Contract A gets a completely different "
            "PE vector than at position 3 in Contract B. "
            "This allows the model to distinguish the two contracts even though their "
            "bag-of-words representations would be identical."
        ),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("pe_outputs", exist_ok=True)

    pe = PositionalEncoding(d_model=64, max_len=50, dropout=0.0)
    pe_matrix = pe.get_matrix(seq_len=30, d_model=64)

    plot_positional_encoding(
        pe_matrix,
        save_path="pe_outputs/pe_heatmap.png",
        positions_to_highlight=[0, 1, 2, 3, 4]
    )
    plot_per_position_heatmaps(
        pe_matrix, num_positions=5,
        save_path="pe_outputs/per_position.png"
    )

    result = analyse_word_order(save_path="pe_outputs/word_order.png")

    print("\n=== Task 6: Clause Understanding Analysis ===")
    print(f"Contract A: {result['sentence_a']}")
    print(f"Contract B: {result['sentence_b']}")
    print(f"\n{result['explanation']}\n")
    print(f"{'Word':15s} {'PosA':>6} {'PosB':>6} {'PE-diff':>10} {'CosSim':>8}")
    print("-" * 50)
    for word, info in result["word_analysis"].items():
        print(f"{word:15s} {info['position_in_A']:6d} {info['position_in_B']:6d} "
              f"{info['pe_diff_norm']:10.4f} {info['cosine_sim']:8.4f}")
