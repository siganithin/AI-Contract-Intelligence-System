"""
Standalone training script — run this once to train and save models.
Saved models are loaded by the Streamlit app.

Usage:
    python train_models.py
"""

import os
import torch
import pickle
import numpy as np

from task1_eda import build_dataset
from task2_text_engineering import build_pipeline
from task3_baseline_model import train_baseline
from task4_attention_model import train_attention_model

SAVE_DIR   = "saved_models"
SEED       = 2024
N_SAMPLES  = 300
MAX_LEN    = 64
MAX_VOCAB  = 5000
EPOCHS     = 15

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  AI Contract Intelligence System — Training")
    print("=" * 60)

    # ── 1. Build dataset ──────────────────────────────────────
    print(f"\n[1/4] Building dataset ({N_SAMPLES} samples)...")
    df = build_dataset(N_SAMPLES)
    print(f"      Dataset shape: {df.shape}")

    # ── 2. Text engineering ───────────────────────────────────
    print(f"\n[2/4] Running text engineering pipeline...")
    result = build_pipeline(df, max_len=MAX_LEN, max_vocab=MAX_VOCAB)
    for k, v in result["report"].items():
        print(f"      {k:30s}: {v}")

    # Save vocab and pipeline
    with open(os.path.join(SAVE_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(result["vocab"], f)
    with open(os.path.join(SAVE_DIR, "pipeline_result.pkl"), "wb") as f:
        # Don't save large arrays, just vocab and report
        save_pr = {"vocab": result["vocab"], "report": result["report"]}
        pickle.dump(save_pr, f)
    print(f"      Vocabulary saved.")

    # ── 3. Train baseline ─────────────────────────────────────
    print(f"\n[3/4] Training baseline model ({EPOCHS} epochs)...")
    baseline_model, baseline_metrics = train_baseline(result, df, num_epochs=EPOCHS)
    print(f"      Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    torch.save(baseline_model.state_dict(),
               os.path.join(SAVE_DIR, "baseline_model.pt"))
    with open(os.path.join(SAVE_DIR, "baseline_metrics.pkl"), "wb") as f:
        pickle.dump(baseline_metrics, f)
    print(f"      Baseline model saved.")

    # ── 4. Train attention model ──────────────────────────────
    print(f"\n[4/4] Training attention model ({EPOCHS} epochs)...")
    attn_model, attn_metrics = train_attention_model(result, df, num_epochs=EPOCHS)
    print(f"      Attention Accuracy: {attn_metrics['accuracy']:.4f}")
    torch.save(attn_model.state_dict(),
               os.path.join(SAVE_DIR, "attn_model.pt"))
    with open(os.path.join(SAVE_DIR, "attn_metrics.pkl"), "wb") as f:
        pickle.dump(attn_metrics, f)
    print(f"      Attention model saved.")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Baseline Accuracy : {baseline_metrics['accuracy']*100:.1f}%")
    print(f"  Attention Accuracy: {attn_metrics['accuracy']*100:.1f}%")
    print(f"\n  Saved to: {os.path.abspath(SAVE_DIR)}/")
    print("\nRun: streamlit run app.py")


if __name__ == "__main__":
    main()
