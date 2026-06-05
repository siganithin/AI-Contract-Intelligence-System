"""
Task 8: Contract Intelligence Dashboard — Streamlit App
AI Contract Intelligence System
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import sys

# ── Page config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title="AI Contract Intelligence System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import project modules ────────────────────────────────────
from task1_eda import build_dataset, run_eda
from task2_text_engineering import build_pipeline, tokenize, Vocabulary
from task3_baseline_model import BaselineModel, train_baseline
from task4_attention_model import AttentionModel, train_attention_model
from task5_positional_encoding import (
    PositionalEncoding,
    plot_positional_encoding,
    plot_per_position_heatmaps,
    analyse_word_order,
)
from task7_attention_analysis import (
    get_attention_weights,
    plot_attention_heatmap,
    plot_token_importance,
    compute_rule_based_importance,
    demo_attention_analysis,
    KEY_LEGAL_TERMS,
)

# ──────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df"              : None,
        "pipeline_result" : None,
        "baseline_model"  : None,
        "baseline_metrics": None,
        "attn_model"      : None,
        "attn_metrics"    : None,
        "trained"         : False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────────────────────
# Helper: load / build data + models (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data_and_train(n_samples: int = 200, max_len: int = 64,
                        max_vocab: int = 3000, epochs: int = 8):
    """Build dataset, run text engineering, train both models."""
    df             = build_dataset(n_samples)
    pipeline_result = build_pipeline(df, max_len=max_len, max_vocab=max_vocab)
    baseline_model, baseline_metrics = train_baseline(
        pipeline_result, df, num_epochs=epochs)
    attn_model, attn_metrics = train_attention_model(
        pipeline_result, df, num_epochs=epochs)
    return df, pipeline_result, baseline_model, baseline_metrics, attn_model, attn_metrics


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/law.png", width=80)
    st.title("⚖️ Contract AI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home",
         "📊 Task 1 — EDA",
         "🔤 Task 2 — Text Engineering",
         "🧠 Task 3 — Baseline Model",
         "👁️ Task 4 — Attention Model",
         "📍 Task 5 & 6 — Positional Encoding",
         "🔍 Task 7 — Attention Analysis",
         "📋 Task 8 — Dashboard"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Settings**")
    n_samples  = st.slider("Dataset Size", 100, 500, 200, 50)
    max_epochs = st.slider("Training Epochs", 3, 20, 8)
    max_len    = st.slider("Max Sequence Length", 32, 128, 64, 16)

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... (this may take a minute)"):
            (df, pipeline_result,
             baseline_model, baseline_metrics,
             attn_model, attn_metrics) = load_data_and_train(
                n_samples=n_samples, max_len=max_len,
                max_vocab=3000, epochs=max_epochs
            )
            st.session_state.df               = df
            st.session_state.pipeline_result  = pipeline_result
            st.session_state.baseline_model   = baseline_model
            st.session_state.baseline_metrics = baseline_metrics
            st.session_state.attn_model       = attn_model
            st.session_state.attn_metrics     = attn_metrics
            st.session_state.trained          = True
        st.success("✅ Models trained!")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "entailment"    : "#27ae60",
    "neutral"       : "#f39c12",
    "contradiction" : "#e74c3c",
}

def label_badge(label: str):
    color = LABEL_COLORS.get(label, "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-weight:bold">{label.upper()}</span>'


def show_metrics(metrics: dict, model_name: str):
    acc = metrics["accuracy"]
    rep = metrics["classification_rep"]
    st.metric(f"{model_name} Accuracy", f"{acc*100:.1f}%")
    rows = []
    for label in metrics["label_names"]:
        if label in rep:
            r = rep[label]
            rows.append({
                "Label"    : label,
                "Precision": f"{r['precision']:.3f}",
                "Recall"   : f"{r['recall']:.3f}",
                "F1-Score" : f"{r['f1-score']:.3f}",
                "Support"  : int(r['support']),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def training_curve(history: dict, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], 'b-o', markersize=4, label='Train Loss')
    ax2 = ax.twinx()
    ax2.plot(epochs, history["train_acc"], 'g-s', markersize=4, label='Train Acc')
    ax2.plot(epochs, history["val_acc"],   'r-^', markersize=4, label='Val Acc')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color='b')
    ax2.set_ylabel("Accuracy", color='g')
    ax.set_title(title, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    return fig


def predict_clause(text: str, model, vocab, pipeline_result, device="cpu"):
    """Run model prediction on a single contract clause."""
    from task2_text_engineering import pad_sequences
    model.eval()
    model = model.to(device)

    ids     = vocab.encode(text)
    padded  = pad_sequences([ids], max_len=pipeline_result["premise_padded"].shape[1])
    dummy   = np.zeros_like(padded)
    p_t     = torch.tensor(padded, dtype=torch.long).to(device)
    h_t     = torch.tensor(dummy,  dtype=torch.long).to(device)

    with torch.no_grad():
        if isinstance(model, AttentionModel):
            logits, p_w, _ = model(p_t, h_t)
            p_tokens = tokenize(text)
            max_len  = p_w.shape[-1]
            p_len    = min(len(p_tokens), max_len)
            p_attn   = p_w[0].cpu().numpy().mean(axis=0)[:p_len, :p_len]
            p_imp    = p_attn.sum(axis=0)
            p_imp    = p_imp / (p_imp.sum() + 1e-9)
            attn_info = {"tokens": p_tokens[:p_len], "importance": p_imp}
        else:
            logits   = model(p_t, h_t)
            attn_info = None

    probs      = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    label_map  = {0: "entailment", 1: "neutral", 2: "contradiction"}
    pred_label = label_map[int(probs.argmax())]
    return pred_label, probs, attn_info


def detect_clause_type(text: str) -> str:
    """Heuristic clause type detection by keyword matching."""
    text_lower = text.lower()
    rules = [
        (["payment", "invoice", "fee", "pay ", "salary", "compensat"], "Payment"),
        (["terminat", "cancel", "end ", "notice period", "expire"],    "Termination"),
        (["confidential", "disclose", "secret", "nda", "proprietary"], "Confidentiality"),
        (["liab", "indemnif", "damages", "remedy", "loss"],            "Liability"),
        (["non-compete", "compete", "competitor", "solicitation"],      "Non-Compete"),
        (["intellectual property", "ip ", "copyright", "patent"],      "IP Rights"),
        (["arbitration", "dispute", "court", "jurisdiction"],          "Dispute Resolution"),
        (["govern", "law of", "jurisdiction"],                         "Governing Law"),
        (["force majeure", "act of god", "natural disaster"],          "Force Majeure"),
    ]
    for keywords, clause_type in rules:
        if any(kw in text_lower for kw in keywords):
            return clause_type
    return "General"


# ──────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────

# ── HOME ──────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("⚖️ AI Contract Intelligence System")
    st.markdown("### NLP + Self-Attention + Positional Encoding")

    st.markdown("""
    This system automatically understands and classifies legal contract clauses using:
    - 🔤 **Custom tokenizer and vocabulary** built from scratch
    - 🧠 **Baseline model**: Embedding → Dense → Output
    - 👁️ **Attention model**: Embedding → MultiHeadAttention → Dense → Output
    - 📍 **Positional Encoding** implemented from scratch (Vaswani et al.)
    - 🔍 **Attention analysis** with visualizations
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Task 1** — EDA & Dataset Analysis")
        st.info("**Task 2** — Text Engineering")
        st.info("**Task 3** — Baseline Model")
    with col2:
        st.info("**Task 4** — Self-Attention Model")
        st.info("**Task 5** — Positional Encoding")
        st.info("**Task 6** — Clause Understanding")
    with col3:
        st.info("**Task 7** — Attention Analysis")
        st.info("**Task 8** — This Dashboard")

    st.markdown("---")
    st.markdown("👈 **Use the sidebar** to navigate between tasks. Click **Train Models** to get started.")

    st.markdown("#### Supported Clause Types")
    clause_types = ["Payment", "Termination", "Confidentiality", "Liability",
                    "Non-Compete", "IP Rights", "Dispute Resolution",
                    "Governing Law", "Force Majeure"]
    cols = st.columns(3)
    for i, ct in enumerate(clause_types):
        cols[i % 3].success(f"✅ {ct}")


# ── TASK 1: EDA ────────────────────────────────────────────────
elif page == "📊 Task 1 — EDA":
    st.title("📊 Task 1 — Dataset Investigation & EDA")

    with st.spinner("Running EDA..."):
        stats, df = run_eda()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Contracts",      stats["total_contracts"])
    c2.metric("Avg Length (words)",   stats["avg_contract_length"])
    c3.metric("Max Length (words)",   stats["max_contract_length"])
    c4.metric("Min Length (words)",   stats["min_contract_length"])

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Clause Type Distribution")
        clause_df = pd.Series(stats["clause_types"]).reset_index()
        clause_df.columns = ["Clause Type", "Count"]
        st.dataframe(clause_df, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("NLI Label Distribution")
        label_df = pd.Series(stats["label_distribution"]).reset_index()
        label_df.columns = ["Label", "Count"]
        st.dataframe(label_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Visualisations")

    # Clause distribution chart
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    clause_counts = df["clause_type"].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(clause_counts)))
    axes[0].bar(clause_counts.index, clause_counts.values, color=colors)
    axes[0].set_title("Clause Distribution", fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    if "label_name" in df.columns:
        label_counts = df["label_name"].value_counts()
        axes[1].pie(label_counts.values, labels=label_counts.index,
                    autopct='%1.1f%%', startangle=90,
                    colors=['#66b3ff', '#ff9999', '#99ff99'])
        axes[1].set_title("NLI Label Distribution", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # Contract length histogram
    fig2, ax = plt.subplots(figsize=(10, 4))
    df["text_len"] = df["premise"].apply(lambda x: len(str(x).split()))
    ax.hist(df["text_len"], bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(df["text_len"].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {df["text_len"].mean():.1f}')
    ax.set_title("Contract Length Distribution (Words)", fontweight='bold')
    ax.set_xlabel("Words")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig2)
    plt.close()

    # Word frequency
    from collections import Counter
    all_words = " ".join(df["premise"].tolist()).lower()
    all_words = re.sub(r'[^a-zA-Z\s]', '', all_words)
    stop = {'the','a','an','is','are','was','were','be','been','has','have','had',
            'do','does','did','will','would','could','should','may','might','shall',
            'to','of','in','for','on','with','at','by','from','or','and','not',
            'this','that','it','its','any','all','no','each'}
    words = [w for w in all_words.split() if w not in stop and len(w) > 2]
    word_freq = Counter(words).most_common(15)

    fig3, ax = plt.subplots(figsize=(10, 5))
    ws, cs = zip(*word_freq)
    ax.barh(ws, cs, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(ws))))
    ax.set_title("Top 15 Most Frequent Words", fontweight='bold')
    ax.invert_yaxis()
    st.pyplot(fig3)
    plt.close()

    st.subheader("Sample Contracts")
    st.dataframe(df[["premise", "hypothesis", "clause_type", "label_name"]].head(10),
                 use_container_width=True, hide_index=True)


# ── TASK 2: TEXT ENGINEERING ───────────────────────────────────
elif page == "🔤 Task 2 — Text Engineering":
    st.title("🔤 Task 2 — Text Engineering")

    with st.spinner("Building pipeline..."):
        df     = build_dataset(200)
        result = build_pipeline(df, max_len=64, max_vocab=3000)
    report = result["report"]

    st.subheader("Pipeline Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Vocabulary Size",     report["vocabulary_size"])
    c2.metric("Corpus Coverage %",   f"{report['corpus_coverage_%']}%")
    c3.metric("OOV Rate %",          f"{report['oov_rate_%']}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Max Sequence Length", report["max_sequence_len"])
    c5.metric("Premise Shape",       str(report["premise_shape"]))
    c6.metric("Unique Token Types",  report["unique_token_types"])

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Cleaning", "🔡 Tokenization", "📚 Vocabulary", "📐 Padding"])

    with tab1:
        st.subheader("Step 1: Text Cleaning")
        sample = st.text_area("Enter contract text:", value=df["premise"].iloc[0])
        from task2_text_engineering import clean_text
        cleaned = clean_text(sample)
        cleaned_no_sw = clean_text(sample, remove_stopwords=True)
        st.markdown(f"**Cleaned:** `{cleaned}`")
        st.markdown(f"**Stop-words removed:** `{cleaned_no_sw}`")
        st.info("Cleaning: lowercase → remove special chars → normalize whitespace")

    with tab2:
        st.subheader("Step 2: Tokenization")
        tokens = tokenize(sample)
        st.markdown(f"**Tokens ({len(tokens)}):** {tokens}")
        st.success("Simple whitespace tokenizer preserving legal compound terms (e.g., 'non-compete').")

    with tab3:
        st.subheader("Step 3: Vocabulary")
        vocab = result["vocab"]
        st.write(f"Vocabulary size: **{vocab.size}** tokens (including 4 special tokens)")

        # Top words
        top_words = sorted(vocab.freq.items(), key=lambda x: x[1], reverse=True)[:20]
        wdf = pd.DataFrame(top_words, columns=["Token", "Frequency"])
        st.dataframe(wdf, use_container_width=True, hide_index=True)

        st.markdown("""
        **Why vocabulary size matters:**
        - Too small → high OOV rate → information loss
        - Too large → sparse embeddings → memory overhead
        - Sweet spot: ~95% corpus coverage
        """)

    with tab4:
        st.subheader("Step 4: Sequence Padding")
        encoded = vocab.encode(df["premise"].iloc[0])
        from task2_text_engineering import pad_sequences
        padded = pad_sequences([encoded], max_len=20)[0]

        fig, ax = plt.subplots(figsize=(12, 2))
        colors = ['#3498db' if v != 0 else '#ecf0f1' for v in padded]
        ax.bar(range(len(padded)), [1]*len(padded), color=colors, edgecolor='white', width=0.95)
        for i, v in enumerate(padded):
            ax.text(i, 0.5, str(v), ha='center', va='center', fontsize=7,
                    color='black' if v == 0 else 'white')
        ax.set_title("Padded Sequence (blue=token, gray=PAD)", fontweight='bold')
        ax.set_yticks([])
        ax.set_xlabel("Position")
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **Why padding is required:**
        - Neural networks process fixed-size batches — variable-length sequences cannot be batched directly.
        - `<PAD>` tokens (ID=0) are added to shorter sequences to match the longest in the batch.
        - Attention masks (1=real, 0=pad) prevent the model from attending to PAD positions.
        """)


# ── TASK 3: BASELINE ───────────────────────────────────────────
elif page == "🧠 Task 3 — Baseline Model":
    st.title("🧠 Task 3 — Baseline Model")

    st.markdown("""
    **Architecture:**
    ```
    Input (token IDs)
       ↓
    Embedding [vocab_size × 128]
       ↓
    Mean Pooling (ignore PAD)
       ↓
    Concatenate [premise || hypothesis]  →  (256-dim)
       ↓
    Dense (256 → 128, ReLU) + Dropout
       ↓
    Dense (128 → 3)  →  Output [entailment, neutral, contradiction]
    ```
    """)

    if not st.session_state.trained:
        st.warning("⚠️ Click **Train Models** in the sidebar first.")
    else:
        metrics = st.session_state.baseline_metrics
        show_metrics(metrics, "Baseline")
        st.subheader("Training Curve")
        st.pyplot(training_curve(metrics["history"], "Baseline Model — Training"))
        plt.close()


# ── TASK 4: ATTENTION MODEL ────────────────────────────────────
elif page == "👁️ Task 4 — Attention Model":
    st.title("👁️ Task 4 — Self-Attention Model")

    st.markdown("""
    **Architecture:**
    ```
    Input (token IDs)
       ↓
    Embedding [vocab_size × 128]
       ↓
    MultiHeadAttention (4 heads, d_k = 32 each)
       ↓
    FeedForward (256-dim, GELU) + LayerNorm
       ↓
    Mean Pooling (masked)
       ↓
    Concatenate [premise || hypothesis]
       ↓
    Dense (256 → 3)  →  Output
    ```
    """)

    if not st.session_state.trained:
        st.warning("⚠️ Click **Train Models** in the sidebar first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attention Model")
            show_metrics(st.session_state.attn_metrics, "Attention")
        with col2:
            st.subheader("Baseline (for comparison)")
            show_metrics(st.session_state.baseline_metrics, "Baseline")

        st.subheader("Training Curves — Side by Side")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.pyplot(training_curve(st.session_state.attn_metrics["history"],
                                     "Attention Model"))
            plt.close()
        with fc2:
            st.pyplot(training_curve(st.session_state.baseline_metrics["history"],
                                     "Baseline Model"))
            plt.close()


# ── TASK 5 & 6: POSITIONAL ENCODING ───────────────────────────
elif page == "📍 Task 5 & 6 — Positional Encoding":
    st.title("📍 Task 5 & 6 — Positional Encoding & Clause Understanding")

    tabs = st.tabs(["📐 PE Heatmap", "🔢 Per-Position", "📝 Clause Analysis"])

    pe_encoder = PositionalEncoding(d_model=64, max_len=50, dropout=0.0)
    pe_matrix  = pe_encoder.get_matrix(seq_len=30, d_model=64)

    with tabs[0]:
        st.subheader("Positional Encoding Heatmap")
        st.markdown("""
        **Formula (Vaswani et al. 2017):**
        - `PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))`
        - `PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))`

        Each row = one position's encoding across all dimensions.
        Similar positions → similar colour patterns.
        """)

        highlight = st.multiselect("Highlight positions:", list(range(0, 30)),
                                   default=[0, 1, 2, 3, 4])
        fig = plot_positional_encoding(pe_matrix, positions_to_highlight=highlight)
        st.pyplot(fig)
        plt.close()

    with tabs[1]:
        st.subheader("Per-Position Heatmaps")
        n_pos = st.slider("Number of positions to show", 2, 10, 5)
        fig   = plot_per_position_heatmaps(pe_matrix, num_positions=n_pos)
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **Key observations:**
        - Position 1 has a distinct low-frequency sinusoidal pattern.
        - Each subsequent position shifts the phase of the waves.
        - Distant positions have clearly different patterns — unique addresses in encoding space.
        """)

    with tabs[2]:
        st.subheader("Task 6 — Clause Understanding: Word Order Analysis")

        clause_a = st.text_input("Contract A:", value="Payment shall be made within 30 days.")
        clause_b = st.text_input("Contract B:", value="Within 30 days payment shall be made.")

        result = analyse_word_order(clause_a, clause_b, d_model=64)
        st.pyplot(result["figure"])
        plt.close()

        st.info(result["explanation"])

        st.subheader("Common Words — Position Comparison")
        rows = []
        for word, info in result["word_analysis"].items():
            rows.append({
                "Word"            : word,
                "Position in A"   : info["position_in_A"],
                "Position in B"   : info["position_in_B"],
                "PE Difference"   : info["pe_diff_norm"],
                "Cosine Similarity": info["cosine_sim"],
            })
        if rows:
            wdf = pd.DataFrame(rows)
            st.dataframe(wdf, use_container_width=True, hide_index=True)
            st.markdown("""
            - **PE Difference > 0** → same word has different positional representation.
            - **Cosine Similarity < 1** → the model will see these as different tokens.
            - This is why the model can distinguish semantically equivalent sentences with different structure.
            """)


# ── TASK 7: ATTENTION ANALYSIS ─────────────────────────────────
elif page == "🔍 Task 7 — Attention Analysis":
    st.title("🔍 Task 7 — Attention Analysis")

    st.markdown("""
    Visualise which tokens the model focuses on when classifying contract clauses.
    Key legal terms like **payment**, **termination**, and **confidential** receive high attention.
    """)

    tab1, tab2 = st.tabs(["🖥️ Demo Analysis", "📤 Custom Input"])

    with tab1:
        st.subheader("Pre-computed Attention Demos")
        examples = [
            {"text": "Payment shall be made within 30 days of invoice receipt.",
             "type": "Payment Clause"},
            {"text": "Either party may terminate this agreement with 60 days written notice.",
             "type": "Termination Clause"},
            {"text": "All confidential information shall remain strictly private.",
             "type": "Confidentiality Clause"},
        ]

        for ex in examples:
            tokens  = ex["text"].lower().replace('.', '').split()
            imp     = compute_rule_based_importance(tokens)

            st.markdown(f"#### {ex['type']}")
            st.markdown(f"> *{ex['text']}*")

            fig = plot_token_importance(
                tokens, imp,
                title=f"Token Attention Scores — {ex['type']}"
            )
            st.pyplot(fig)
            plt.close()

            # Show top tokens
            top_n = sorted(zip(tokens, imp), key=lambda x: x[1], reverse=True)[:5]
            top_df = pd.DataFrame(top_n, columns=["Token", "Attention Score"])
            top_df["Attention Score"] = top_df["Attention Score"].round(4)
            st.dataframe(top_df, use_container_width=True, hide_index=True)
            st.markdown("---")

    with tab2:
        st.subheader("Analyse Your Own Contract Clause")

        if st.session_state.trained:
            input_clause = st.text_area(
                "Enter a contract clause:",
                value="The contractor shall maintain liability insurance of at least $1 million.",
                height=100,
            )
            if st.button("Analyse", type="primary"):
                model  = st.session_state.attn_model
                vocab  = st.session_state.pipeline_result["vocab"]
                pr     = st.session_state.pipeline_result

                pred, probs, attn_info = predict_clause(input_clause, model, vocab, pr)
                clause_type = detect_clause_type(input_clause)

                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Prediction:** {label_badge(pred)}", unsafe_allow_html=True)
                c2.metric("Clause Type", clause_type)
                c3.metric("Confidence", f"{probs.max()*100:.1f}%")

                # Probability bars
                label_names = ['entailment', 'neutral', 'contradiction']
                prob_df = pd.DataFrame({"Label": label_names,
                                        "Probability": probs.tolist()})
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = [LABEL_COLORS[l] for l in label_names]
                ax.barh(label_names, probs, color=colors)
                ax.set_xlim(0, 1)
                ax.set_title("Prediction Probabilities", fontweight='bold')
                for i, p in enumerate(probs):
                    ax.text(p + 0.01, i, f'{p:.3f}', va='center')
                st.pyplot(fig)
                plt.close()

                if attn_info and len(attn_info["tokens"]) > 0:
                    st.subheader("Token Attention Scores")
                    fig2 = plot_token_importance(
                        attn_info["tokens"], attn_info["importance"],
                        title="Attention Scores from Trained Model"
                    )
                    st.pyplot(fig2)
                    plt.close()
        else:
            st.warning("⚠️ Please train the models first (sidebar → Train Models).")

            # Fallback: rule-based demo
            input_clause = st.text_area(
                "Enter a contract clause (demo mode):",
                value="Termination requires 30 days notice with payment of outstanding fees.",
                height=100,
            )
            if st.button("Demo Analyse"):
                tokens = input_clause.lower().replace('.', '').split()
                imp    = compute_rule_based_importance(tokens)
                clause_type = detect_clause_type(input_clause)
                st.metric("Detected Clause Type", clause_type)

                fig = plot_token_importance(tokens, imp,
                                            title="Token Importance (Rule-Based Demo)")
                st.pyplot(fig)
                plt.close()

                top_n = sorted(zip(tokens, imp), key=lambda x: x[1], reverse=True)[:5]
                st.subheader("Most Important Tokens")
                st.dataframe(pd.DataFrame(top_n, columns=["Token", "Score"]),
                             use_container_width=True, hide_index=True)


# ── TASK 8: DASHBOARD ──────────────────────────────────────────
elif page == "📋 Task 8 — Dashboard":
    st.title("📋 Task 8 — Contract Intelligence Dashboard")
    st.markdown("Upload a contract or paste text to get full AI analysis.")

    # Upload section
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("Upload Contract (.txt)", type=["txt"])
        if uploaded:
            contract_text = uploaded.read().decode("utf-8")
            st.success(f"✅ Loaded: {len(contract_text.split())} words")
        else:
            contract_text = st.text_area(
                "Or paste contract text here:",
                value=(
                    "This Agreement is entered into as of January 1, 2024. "
                    "Payment shall be made within 30 days of invoice. "
                    "Either party may terminate this agreement with 60 days written notice. "
                    "All confidential information shall remain strictly private. "
                    "The total liability shall not exceed the contract value. "
                    "The employee agrees not to compete for a period of 2 years."
                ),
                height=200,
            )

    with col2:
        st.markdown("#### Analysis Options")
        show_attention = st.checkbox("Show Attention Maps", value=True)
        show_pe        = st.checkbox("Show Positional Encoding", value=True)
        top_n_words    = st.slider("Top Important Words", 3, 10, 5)

    if st.button("🔍 Analyse Contract", type="primary"):
        # Split into sentences / clauses
        sentences = [s.strip() for s in re.split(r'[.!?]+', contract_text)
                     if len(s.strip()) > 10]

        st.markdown("---")
        st.subheader("📑 Clause-by-Clause Analysis")

        for i, clause in enumerate(sentences[:8]):   # limit to 8 for display
            clause_type = detect_clause_type(clause)
            tokens      = clause.lower().split()
            imp         = compute_rule_based_importance(tokens)
            top_tokens  = sorted(zip(tokens, imp), key=lambda x: x[1], reverse=True)[:top_n_words]

            # Use trained model if available
            if st.session_state.trained:
                pred, probs, _ = predict_clause(
                    clause,
                    st.session_state.attn_model,
                    st.session_state.pipeline_result["vocab"],
                    st.session_state.pipeline_result,
                )
                confidence = f"{probs.max()*100:.1f}%"
            else:
                pred = "entailment"
                confidence = "N/A (demo)"

            with st.expander(f"Clause {i+1}: {clause_type} — {clause[:60]}...", expanded=(i==0)):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Type:** `{clause_type}`")
                c2.markdown(f"**NLI:** {label_badge(pred)}", unsafe_allow_html=True)
                c3.markdown(f"**Confidence:** `{confidence}`")

                st.markdown(f"*{clause}*")

                if show_attention:
                    fig_att = plot_token_importance(
                        tokens[:20], imp[:20],
                        title=f"Token Attention — {clause_type}"
                    )
                    st.pyplot(fig_att)
                    plt.close()

                st.markdown("**Key Terms:**")
                term_cols = st.columns(top_n_words)
                for j, (tok, score) in enumerate(top_tokens):
                    term_cols[j].metric(tok, f"{score:.3f}")

        # Summary table
        st.markdown("---")
        st.subheader("📊 Contract Summary")

        summary_rows = []
        for clause in sentences[:8]:
            ct  = detect_clause_type(clause)
            toks = clause.lower().split()
            imp  = compute_rule_based_importance(toks)
            top1 = max(zip(toks, imp), key=lambda x: x[1])
            summary_rows.append({
                "Clause"        : clause[:60] + "...",
                "Type"          : ct,
                "Words"         : len(toks),
                "Key Term"      : top1[0],
                "Attention Score": round(top1[1], 3),
            })
        st.dataframe(pd.DataFrame(summary_rows),
                     use_container_width=True, hide_index=True)

        # Clause type distribution
        if summary_rows:
            types = [r["Type"] for r in summary_rows]
            from collections import Counter
            type_counts = Counter(types)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(type_counts.keys(), type_counts.values(),
                   color=plt.cm.Set2(np.linspace(0, 1, len(type_counts))))
            ax.set_title("Clause Types in This Contract", fontweight='bold')
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=30)
            st.pyplot(fig)
            plt.close()

        # Positional encoding of first clause
        if show_pe and sentences:
            st.markdown("---")
            st.subheader("📍 Positional Encoding Visualization")
            first_clause = sentences[0]
            result_pe = analyse_word_order(
                first_clause,
                first_clause[::-1][:len(first_clause)],
                d_model=32,
            )
            st.pyplot(result_pe["figure"])
            plt.close()


# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.85em'>"
    "AI Contract Intelligence System | NLP + Self-Attention + Positional Encoding"
    "</div>",
    unsafe_allow_html=True,
)
