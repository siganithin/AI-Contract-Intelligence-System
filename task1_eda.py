"""
Task 1: Dataset Investigation & Exploratory Data Analysis
AI Contract Intelligence System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# ─────────────────────────────────────────────
# Sample contract dataset (used when Kaggle not available)
# ─────────────────────────────────────────────
SAMPLE_CONTRACTS = [
    {
        "premise": "Payment shall be made within 30 days of invoice receipt.",
        "hypothesis": "The company must pay within one month.",
        "label": "e",
        "clause_type": "Payment"
    },
    {
        "premise": "Either party may terminate this agreement with 60 days written notice.",
        "hypothesis": "The contract can be ended by both parties.",
        "label": "e",
        "clause_type": "Termination"
    },
    {
        "premise": "All confidential information shall be kept strictly private.",
        "hypothesis": "The company may share confidential data freely.",
        "label": "c",
        "clause_type": "Confidentiality"
    },
    {
        "premise": "The total liability shall not exceed the contract value.",
        "hypothesis": "There is no limit on liability.",
        "label": "c",
        "clause_type": "Liability"
    },
    {
        "premise": "The employee agrees not to compete for a period of 2 years.",
        "hypothesis": "The employee is restricted from joining competitors.",
        "label": "e",
        "clause_type": "Non-Compete"
    },
    {
        "premise": "The governing law shall be the law of New York State.",
        "hypothesis": "Disputes will be resolved under New York law.",
        "label": "e",
        "clause_type": "Governing Law"
    },
    {
        "premise": "All intellectual property created during employment belongs to the employer.",
        "hypothesis": "Employees retain ownership of their creations.",
        "label": "c",
        "clause_type": "IP Rights"
    },
    {
        "premise": "Services shall be provided as described in Schedule A.",
        "hypothesis": "The scope of work is defined separately.",
        "label": "e",
        "clause_type": "Scope of Work"
    },
    {
        "premise": "The agreement shall automatically renew unless terminated 30 days prior.",
        "hypothesis": "The contract needs manual renewal each year.",
        "label": "c",
        "clause_type": "Termination"
    },
    {
        "premise": "Confidential information shall not be disclosed without prior written consent.",
        "hypothesis": "No information can be shared without written approval.",
        "label": "e",
        "clause_type": "Confidentiality"
    },
    {
        "premise": "The contractor shall indemnify the company against any third-party claims.",
        "hypothesis": "The company bears no responsibility for contractor's actions.",
        "label": "e",
        "clause_type": "Liability"
    },
    {
        "premise": "Payment terms are net 45 days from delivery.",
        "hypothesis": "Payment is due immediately upon delivery.",
        "label": "c",
        "clause_type": "Payment"
    },
    {
        "premise": "Either party may request additional services at agreed rates.",
        "hypothesis": "No additional services can be requested.",
        "label": "c",
        "clause_type": "Scope of Work"
    },
    {
        "premise": "The non-compete clause applies only within the United States.",
        "hypothesis": "The restriction is limited to the US.",
        "label": "e",
        "clause_type": "Non-Compete"
    },
    {
        "premise": "Force majeure events shall excuse performance obligations.",
        "hypothesis": "Natural disasters may suspend contract obligations.",
        "label": "e",
        "clause_type": "Force Majeure"
    },
    {
        "premise": "Disputes shall be resolved through binding arbitration.",
        "hypothesis": "Parties must go to court to resolve disputes.",
        "label": "c",
        "clause_type": "Dispute Resolution"
    },
    {
        "premise": "The confidentiality obligation survives termination of the agreement.",
        "hypothesis": "Confidentiality ends when the contract ends.",
        "label": "c",
        "clause_type": "Confidentiality"
    },
    {
        "premise": "The contractor shall maintain professional liability insurance.",
        "hypothesis": "Insurance requirements are optional.",
        "label": "c",
        "clause_type": "Liability"
    },
    {
        "premise": "Invoices must be submitted monthly by the last business day.",
        "hypothesis": "Billing is done on a monthly basis.",
        "label": "e",
        "clause_type": "Payment"
    },
    {
        "premise": "Either party may terminate for cause with immediate effect.",
        "hypothesis": "Immediate termination is allowed for serious breaches.",
        "label": "e",
        "clause_type": "Termination"
    },
    {
        "premise": "The software license is non-exclusive and non-transferable.",
        "hypothesis": "The license can be transferred to third parties.",
        "label": "c",
        "clause_type": "IP Rights"
    },
    {
        "premise": "Annual salary shall be reviewed each January.",
        "hypothesis": "Salary reviews happen once a year.",
        "label": "e",
        "clause_type": "Payment"
    },
    {
        "premise": "The contractor shall not subcontract without written consent.",
        "hypothesis": "Subcontracting requires approval.",
        "label": "e",
        "clause_type": "Scope of Work"
    },
    {
        "premise": "This agreement constitutes the entire understanding between the parties.",
        "hypothesis": "Other agreements may also be valid.",
        "label": "c",
        "clause_type": "Governing Law"
    },
    {
        "premise": "The company shall provide a 30-day cure period before termination.",
        "hypothesis": "Termination can occur without any notice period.",
        "label": "c",
        "clause_type": "Termination"
    },
]

# Extend sample to simulate larger dataset
def build_dataset(n=200):
    """Build a representative sample dataset with numeric labels."""
    base = SAMPLE_CONTRACTS.copy()
    extended = []
    str_to_name = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}
    str_to_int  = {'e': 0, 'n': 1, 'c': 2}
    for i in range(n):
        item = base[i % len(base)].copy()
        extra = " " + " ".join(["and"] * (i % 5))
        item["premise"]       = item["premise"] + extra
        item["label_name"]    = str_to_name[item["label"]]
        item["label"]         = str_to_int[item["label"]]   # ← numeric
        item["premise_len"]   = len(item["premise"].split())
        item["hypothesis_len"]= len(item["hypothesis"].split())
        extended.append(item)
    return pd.DataFrame(extended)


def run_eda(df=None, save_dir="eda_outputs"):
    """Run full EDA and return stats dict + save figures."""
    os.makedirs(save_dir, exist_ok=True)

    if df is None:
        df = build_dataset(200)

    df["text_len"] = df["premise"].apply(lambda x: len(str(x).split()))

    # ── Basic Stats ──────────────────────────────────────
    stats = {
        "total_contracts": len(df),
        "clause_types": df["clause_type"].value_counts().to_dict(),
        "label_distribution": df["label_name"].value_counts().to_dict() if "label_name" in df.columns else {},
        "avg_contract_length": round(df["text_len"].mean(), 2),
        "max_contract_length": int(df["text_len"].max()),
        "min_contract_length": int(df["text_len"].min()),
        "longest_contract_idx": int(df["text_len"].idxmax()),
        "shortest_contract_idx": int(df["text_len"].idxmin()),
    }

    # ── Plot 1: Clause Distribution ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    clause_counts = df["clause_type"].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(clause_counts)))
    axes[0].bar(clause_counts.index, clause_counts.values, color=colors)
    axes[0].set_title("Clause Type Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Clause Type")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(axes[0].patches, clause_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha='center', fontsize=9)

    # Label distribution pie
    if "label_name" in df.columns:
        label_counts = df["label_name"].value_counts()
        axes[1].pie(label_counts.values, labels=label_counts.index,
                    autopct='%1.1f%%', startangle=90,
                    colors=['#66b3ff', '#ff9999', '#99ff99'])
        axes[1].set_title("NLI Label Distribution", fontsize=14, fontweight='bold')

    plt.tight_layout()
    path1 = os.path.join(save_dir, "clause_distribution.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Contract Length Histogram ─────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["text_len"], bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(df["text_len"].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {df["text_len"].mean():.1f}')
    ax.set_title("Contract Length Distribution (Words)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    ax.legend()
    path2 = os.path.join(save_dir, "length_histogram.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Word Frequency ────────────────────────────
    all_words = " ".join(df["premise"].tolist()).lower()
    all_words = re.sub(r'[^a-zA-Z\s]', '', all_words)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'shall', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and', 'not',
                  'this', 'that', 'it', 'its', 'any', 'all', 'no', 'each'}
    words = [w for w in all_words.split() if w not in stop_words and len(w) > 2]
    word_freq = Counter(words).most_common(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    words_list, counts = zip(*word_freq)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(words_list)))
    bars = ax.barh(words_list, counts, color=colors)
    ax.set_title("Top 20 Most Frequent Words in Contracts", fontsize=14, fontweight='bold')
    ax.set_xlabel("Frequency")
    ax.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=9)
    path3 = os.path.join(save_dir, "word_frequency.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 4: Clause length by type ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    clause_groups = [df[df["clause_type"] == ct]["text_len"].tolist()
                     for ct in clause_counts.index]
    bp = ax.boxplot(clause_groups, labels=clause_counts.index, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title("Contract Length Distribution by Clause Type", fontsize=14, fontweight='bold')
    ax.set_xlabel("Clause Type")
    ax.set_ylabel("Word Count")
    ax.tick_params(axis='x', rotation=45)
    path4 = os.path.join(save_dir, "length_by_clause.png")
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()

    stats["figure_paths"] = [path1, path2, path3, path4]
    return stats, df


if __name__ == "__main__":
    stats, df = run_eda()
    print("\n=== EDA Summary ===")
    print(f"Total Contracts     : {stats['total_contracts']}")
    print(f"Avg Contract Length : {stats['avg_contract_length']} words")
    print(f"Max Contract Length : {stats['max_contract_length']} words")
    print(f"Min Contract Length : {stats['min_contract_length']} words")
    print(f"\nClause Types:\n{pd.Series(stats['clause_types'])}")
    print(f"\nLabel Distribution:\n{pd.Series(stats['label_distribution'])}")
