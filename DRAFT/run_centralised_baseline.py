"""
run_centralised_baseline.py
----------------------------
Trains DistilBERT and TinyBERT in a standard centralised setting
(no federation, all data available) for direct comparison with
federated results.

This establishes the performance ceiling — how well each model
performs when there are no Non-IID or communication constraints.

Results saved to: results/centralised_baseline.json
                  results/centralised_baseline_rounds.csv

Usage:
  python run_centralised_baseline.py
"""

import os
import sys
import json
import time
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.preprocess import load_and_merge, split_dataset
from models.transformer_classifier import (
    get_model, EmailDataset, evaluate, DEVICE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.RESULTS_DIR, "centralised_log.txt")),
    ]
)
logger = logging.getLogger(__name__)


def train_centralised(
    model_key:  str,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    num_epochs: int   = 10,
    batch_size: int   = 16,
    lr:         float = 2e-5,
    patience:   int   = 3,
) -> dict:
    """
    Standard centralised training loop.
    Trains on full training set, evaluates on val set each epoch,
    applies early stopping, returns test set metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CENTRALISED BASELINE — {model_key.upper()}")
    logger.info(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"{'='*60}")

    start = time.time()
    model, tokenizer = get_model(model_key)

    train_dataset = EmailDataset(
        {"text": train_df["text"].tolist(), "label": train_df["label"].tolist()},
        tokenizer
    )
    val_dataset = EmailDataset(
        {"text": val_df["text"].tolist(), "label": val_df["label"].tolist()},
        tokenizer
    )
    test_dataset = EmailDataset(
        {"text": test_df["text"].tolist(), "label": test_df["label"].tolist()},
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    epoch_results = []
    best_f1       = -1.0
    best_epoch    = 0
    no_improve    = 0

    for epoch in range(1, num_epochs + 1):
        # ── Training ────────────────────────────────────────────────────
        model.train()
        batch_losses = []

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # ── Validation ───────────────────────────────────────────────────
        val_metrics = evaluate(model, val_dataset, batch_size=32)

        record = {
            "model":       model_key,
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1":      val_metrics["f1"],
            "val_auc_roc": val_metrics["auc_roc"],
        }
        epoch_results.append(record)

        logger.info(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc_roc']:.4f}"
        )

        # ── Early stopping ────────────────────────────────────────────────
        if val_metrics["f1"] > best_f1 + config.EARLY_STOP_DELTA:
            best_f1    = val_metrics["f1"]
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

        torch.cuda.empty_cache()

    # ── Final test evaluation ─────────────────────────────────────────────
    test_metrics = evaluate(model, test_dataset, batch_size=32)
    runtime = round(time.time() - start, 1)

    result = {
        "model":            model_key,
        "setting":          "centralised",
        "num_epochs":       len(epoch_results),
        "best_epoch":       best_epoch,
        "accuracy":         test_metrics["accuracy"],
        "f1":               test_metrics["f1"],
        "precision":        test_metrics["precision"],
        "recall":           test_metrics["recall"],
        "auc_roc":          test_metrics["auc_roc"],
        "loss":             test_metrics["loss"],
        "train_size":       len(train_df),
        "runtime_sec":      runtime,
    }

    logger.info(
        f"\n[CENTRALISED {model_key.upper()}] DONE | "
        f"acc={result['accuracy']:.4f} | "
        f"f1={result['f1']:.4f} | "
        f"auc={result['auc_roc']:.4f} | "
        f"time={runtime}s"
    )

    del model
    torch.cuda.empty_cache()

    return result, pd.DataFrame(epoch_results)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    logger.info("Loading and merging datasets...")
    df = load_and_merge()
    train_df, val_df, test_df = split_dataset(df)

    all_results  = []
    all_epoch_dfs = []

    for model_key in ["distilbert", "tinybert"]:
        result, epoch_df = train_centralised(
            model_key  = model_key,
            train_df   = train_df,
            val_df     = val_df,
            test_df    = test_df,
            num_epochs = 10,
            batch_size = config.LOCAL_BATCH,
            lr         = config.LEARNING_RATE,
            patience   = 3,
        )
        all_results.append(result)
        all_epoch_dfs.append(epoch_df)

    # ── Save results ──────────────────────────────────────────────────────
    results_path = os.path.join(config.RESULTS_DIR, "centralised_baseline.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    epochs_path = os.path.join(config.RESULTS_DIR, "centralised_baseline_epochs.csv")
    pd.concat(all_epoch_dfs, ignore_index=True).to_csv(epochs_path, index=False)
    logger.info(f"Epoch logs saved to {epochs_path}")

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "="*70)
    print("CENTRALISED BASELINE RESULTS")
    print("="*70)
    df_results = pd.DataFrame(all_results)
    print(df_results[["model", "accuracy", "f1", "precision",
                       "recall", "auc_roc", "num_epochs", "runtime_sec"]].to_string(index=False))
    print("="*70)

    # ── Compare with federated results ────────────────────────────────────
    fed_summary = os.path.join(config.RESULTS_DIR, "all_experiments_summary.csv")
    if os.path.exists(fed_summary):
        fed_df = pd.read_csv(fed_summary)
        fed_df["f1"] = pd.to_numeric(fed_df["f1"], errors="coerce")

        print("\n" + "="*70)
        print("FEDERATED vs CENTRALISED COMPARISON")
        print("="*70)

        for model_key in ["distilbert", "tinybert"]:
            cent = next((r for r in all_results if r["model"] == model_key), None)
            if cent is None:
                continue

            fed_iid = fed_df[
                (fed_df["model"] == model_key) &
                (fed_df["non_iid_type"] == "label_skew_iid")
            ]["f1"].mean()

            fed_severe = fed_df[
                (fed_df["model"] == model_key) &
                (fed_df["non_iid_type"] == "label_skew_severe")
            ]["f1"].mean()

            fed_worst = fed_df[
                (fed_df["model"] == model_key)
            ]["f1"].min()

            print(f"\n{model_key.upper()}")
            print(f"  Centralised (full data):        F1 = {cent['f1']:.4f}")
            print(f"  Federated IID baseline:         F1 = {fed_iid:.4f}  (gap: {cent['f1']-fed_iid:+.4f})")
            print(f"  Federated label skew severe:    F1 = {fed_severe:.4f}  (gap: {cent['f1']-fed_severe:+.4f})")
            print(f"  Federated worst case:           F1 = {fed_worst:.4f}  (gap: {cent['f1']-fed_worst:+.4f})")

        print("="*70)
        print("\nInterpretation:")
        print("  Small gap = FL successfully preserves centralised performance")
        print("  Large gap = Non-IID data hurts federated training significantly")


if __name__ == "__main__":
    main()
