# ⚖️ AI Contract Intelligence System
### NLP + Self-Attention + Positional Encoding

A complete AI system that understands legal contracts and automatically identifies critical clauses.

## 📦 Project Structure

```
├── app.py                      # Task 8 — Streamlit Dashboard (main entry)
├── task1_eda.py                # Task 1 — Dataset Investigation & EDA
├── task2_text_engineering.py   # Task 2 — Cleaning, Tokenization, Vocab, Padding
├── task3_baseline_model.py     # Task 3 — Baseline Model (Embedding → Dense)
├── task4_attention_model.py    # Task 4 — Self-Attention Model
├── task5_positional_encoding.py # Task 5 & 6 — PE from scratch + Clause Analysis
├── task7_attention_analysis.py # Task 7 — Attention Visualization
├── train_models.py             # Standalone training script
├── requirements.txt            # Dependencies
└── .streamlit/config.toml     # Streamlit theme config
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

### 3. (Optional) Pre-train models
```bash
python train_models.py
```

## 📋 Tasks Implemented

| Task | Description | File |
|------|-------------|------|
| **1** | Dataset EDA — total contracts, clause types, avg/max/min length, visualizations | `task1_eda.py` |
| **2** | Text Engineering — cleaning, tokenization, vocabulary, padding, OOV handling | `task2_text_engineering.py` |
| **3** | Baseline Model — Input → Embedding → Dense → Output | `task3_baseline_model.py` |
| **4** | Attention Model — Input → Embedding → MultiHeadAttention → Dense → Output | `task4_attention_model.py` |
| **5** | Positional Encoding from scratch with heatmap visualizations | `task5_positional_encoding.py` |
| **6** | Clause Understanding — same words, different order, different PE representations | `task5_positional_encoding.py` |
| **7** | Attention Analysis — attention scores, important words (payment/termination/confidential) | `task7_attention_analysis.py` |
| **8** | Streamlit Dashboard — upload contract, predict clause, highlight terms, attention maps | `app.py` |

## 🏗️ Model Architectures

### Baseline Model (Task 3)
```
Input (premise + hypothesis token IDs)
  ↓
Embedding Layer  [vocab_size × 128]
  ↓
Mean Pooling     [batch × 128]
  ↓
Concatenate      [batch × 256]
  ↓
Dense (ReLU)     [256 → 128]
  ↓
Dropout (0.3)
  ↓
Dense (Output)   [128 → 3 classes]
```

### Attention Model (Task 4)
```
Input (token IDs)
  ↓
Embedding [vocab_size × 128]
  ↓
MultiHeadAttention (4 heads, scaled dot-product)
  ↓
FeedForward (GELU) + LayerNorm + Residual
  ↓
Mean Pooling (masked, ignore PAD)
  ↓
Concatenate [premise || hypothesis]
  ↓
Dense → Output [3 classes]
```

## 📍 Positional Encoding (Task 5)

Implemented from scratch using the Vaswani et al. (2017) formulas:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

## 📝 Task 6 — Clause Understanding

**Contract A:** "Payment shall be made within 30 days."
**Contract B:** "Within 30 days payment shall be made."

Same words, different order → different positional representations.
The word "payment" at position 1 (Contract A) has a completely different PE vector
than "payment" at position 3 (Contract B), enabling the model to distinguish clause structure.

## 🔍 Task 7 — Key Legal Terms

The attention model focuses on domain-specific legal keywords:
- **payment** — payment clauses
- **termination** / **terminate** — termination clauses
- **confidential** / **disclose** — confidentiality clauses
- **liability** / **indemnify** — liability clauses
- **compete** — non-compete clauses

## 🌐 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository, set `app.py` as main file
4. Deploy

### Requirements
- Python 3.9+
- PyTorch 2.2+
- Streamlit 1.32+
- See `requirements.txt` for full list

## 📊 Dataset

Based on [kiddothe2b/contract-nli](https://huggingface.co/datasets/kiddothe2b/contract-nli) (ContractNLI-A).
The project includes a built-in sample dataset for offline use and Streamlit deployment.

## 📖 References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- [kiddothe2b/contract-nli](https://www.kaggle.com/code/ahmshalan/contract-nli) — Kaggle notebook
- [Sentence Transformers](https://sbert.net) — cross-encoder NLI models
