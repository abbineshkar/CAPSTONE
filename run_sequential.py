"""
run_sequential.py
-----------------
Lightweight federated learning runner that avoids Ray entirely.
Runs clients one at a time sequentially — much better for 4GB VRAM.

Each round:
  1. Send global weights to client
  2. Client trains locally
  3. Collect updated weights
  4. FedAvg aggregate
  5. Evaluate on validation set
  6. Log metrics

Usage:
  python run_sequential.py
  python run_sequential.py --only E1 E2 E3
  python run_sequential.py --skip_existing
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.preprocess  import load_and_merge, split_dataset
from data.partition   import get_partitions
from models.transformer_classifier import (
    get_model, get_parameters, set_parameters,
    EmailDataset, evaluate, local_train, DEVICE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.RESULTS_DIR, "sequential_run.log")),
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────────────

def fedavg_aggregate(all_params, all_sizes):
    """Weighted average of client parameters by dataset size."""
    total = sum(all_sizes)
    aggregated = []
    for layer_idx in range(len(all_params[0])):
        weighted = sum(
            all_params[c][layer_idx] * (all_sizes[c] / total)
            for c in range(len(all_params))
        )
        aggregated.append(weighted)
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    experiment_id, model_key, aggregation, non_iid_type,
    alpha=None, mu=0.0,
    num_rounds=config.NUM_ROUNDS,
    num_clients=config.NUM_CLIENTS,
):
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT {experiment_id} | {model_key} | {aggregation} | {non_iid_type}")
    logger.info("=" * 60)
    start = time.time()

    # ── Data ──────────────────────────────────────────────────────────────
    df = load_and_merge()
    train_df, val_df, test_df = split_dataset(df)
    partitions = get_partitions(train_df, non_iid_type, alpha=alpha, num_clients=num_clients)

    # ── Build datasets ────────────────────────────────────────────────────
    model, tokenizer = get_model(model_key)

    # Skip empty partitions (can occur under severe Dirichlet skew)
    valid_partitions = [p for p in partitions if len(p) > 0]
    if len(valid_partitions) < len(partitions):
        logger.warning(
            f"Skipping {len(partitions)-len(valid_partitions)} empty client(s) "
            f"— {len(valid_partitions)} active clients this experiment"
        )

    client_datasets = [
        EmailDataset(
            {"text": p["text"].tolist(), "label": p["label"].tolist()},
            tokenizer
        )
        for p in valid_partitions
    ]
    val_dataset  = EmailDataset({"text": val_df["text"].tolist(),  "label": val_df["label"].tolist()},  tokenizer)
    test_dataset = EmailDataset({"text": test_df["text"].tolist(), "label": test_df["label"].tolist()}, tokenizer)

    # ── FL rounds ─────────────────────────────────────────────────────────
    global_params = get_parameters(model)
    round_results = []

    best_f1       = -1.0
    no_improve    = 0

    for rnd in range(1, num_rounds + 1):
        logger.info(f"\n[ROUND {rnd}/{num_rounds}]")
        round_start = time.time()

        all_params = []
        all_sizes  = []
        train_losses = []

        # ── Client training (sequential, one at a time) ──────────────────
        for cid in range(len(client_datasets)):
            logger.info(f"  Client {cid+1}/{num_clients} training...")

            # Load global params into a fresh model instance
            set_parameters(model, global_params)

            # FedProx: snapshot global params before local training
            global_tensors = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

            ds = client_datasets[cid]
            if len(ds) == 0:
                logger.warning(f"  Client {cid+1} has 0 samples — skipping")
                continue

            loader = DataLoader(ds, batch_size=config.LOCAL_BATCH, shuffle=True, num_workers=0)
            optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
            total_steps = len(loader) * config.LOCAL_EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, total_steps // 10),
                num_training_steps=total_steps,
            )

            # Local training epochs
            epoch_losses = []
            for epoch in range(config.LOCAL_EPOCHS):
                model.train()
                batch_losses = []
                for batch in loader:
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss

                    # FedProx proximal term
                    if mu > 0 and global_tensors is not None:
                        prox = sum(
                            torch.norm(lp - gp) ** 2
                            for lp, gp in zip(model.parameters(), global_tensors)
                        )
                        loss = loss + (mu / 2.0) * prox

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    batch_losses.append(loss.item())

                epoch_losses.append(np.mean(batch_losses))

            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            all_params.append(get_parameters(model))
            all_sizes.append(len(ds))
            logger.info(f"  Client {cid+1} done | loss={avg_loss:.4f} | n={len(ds)}")

            # Free GPU memory between clients
            torch.cuda.empty_cache()

        if not all_params:
            logger.warning(f"Round {rnd}: no client updates — skipping")
            continue

        # ── Aggregate ────────────────────────────────────────────────────
        global_params = fedavg_aggregate(all_params, all_sizes)

        # ── Evaluate global model on validation set ───────────────────────
        set_parameters(model, global_params)
        metrics = evaluate(model, val_dataset, batch_size=32)

        record = {
            "experiment_id": experiment_id,
            "round":         rnd,
            "loss":          metrics["loss"],
            "accuracy":      metrics["accuracy"],
            "f1":            metrics["f1"],
            "precision":     metrics["precision"],
            "recall":        metrics["recall"],
            "auc_roc":       metrics["auc_roc"],
            "train_loss":    float(np.mean(train_losses)),
            "round_time_sec": round(time.time() - round_start, 1),
        }
        round_results.append(record)

        logger.info(
            f"[Round {rnd}] {aggregation.upper()} | "
            f"acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} | "
            f"auc={metrics['auc_roc']:.4f} | loss={metrics['loss']:.4f} | "
            f"time={record['round_time_sec']}s"
        )

        # ── Early stopping ────────────────────────────────────────────────
        if metrics["f1"] > best_f1 + config.EARLY_STOP_DELTA:
            best_f1    = metrics["f1"]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_ROUNDS:
                logger.info(f"Early stopping at round {rnd}")
                break

    # ── Final test evaluation ─────────────────────────────────────────────
    set_parameters(model, global_params)
    test_metrics = evaluate(model, test_dataset, batch_size=32)

    # Convergence round
    f1s = [r["f1"] for r in round_results]
    conv_round = next(
        (i+1 for i in range(len(f1s)-config.EARLY_STOP_ROUNDS)
         if all(f1s[i] - f1s[i+j] >= -config.EARLY_STOP_DELTA
                for j in range(1, config.EARLY_STOP_ROUNDS+1))),
        len(f1s)
    )

    final = {
        "experiment_id":    experiment_id,
        "model":            model_key,
        "aggregation":      aggregation,
        "non_iid_type":     non_iid_type,
        "alpha":            alpha,
        "mu":               mu,
        "accuracy":         test_metrics["accuracy"],
        "f1":               test_metrics["f1"],
        "precision":        test_metrics["precision"],
        "recall":           test_metrics["recall"],
        "auc_roc":          test_metrics["auc_roc"],
        "loss":             test_metrics["loss"],
        "convergence_round": conv_round,
        "total_rounds":     len(round_results),
        "runtime_sec":      round(time.time() - start, 1),
    }

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    rounds_path = os.path.join(config.RESULTS_DIR, f"{experiment_id}_rounds.csv")
    pd.DataFrame(round_results).to_csv(rounds_path, index=False)

    final_path = os.path.join(config.RESULTS_DIR, f"{experiment_id}_final.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)

    logger.info(f"[{experiment_id}] ✓ DONE | acc={final['accuracy']:.4f} | f1={final['f1']:.4f} | time={final['runtime_sec']}s")
    logger.info(f"  Saved: {rounds_path}")
    logger.info(f"  Saved: {final_path}")

    # Free memory before next experiment
    del model
    torch.cuda.empty_cache()

    return final


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    matrix = config.EXPERIMENT_MATRIX
    if args.only:
        matrix = [e for e in matrix if e[0] in args.only]

    all_results = []
    failed = []

    for exp_id, model_key, aggregation, non_iid_type, param in matrix:
        final_path = os.path.join(config.RESULTS_DIR, f"{exp_id}_final.json")

        if args.skip_existing and os.path.exists(final_path):
            logger.info(f"[{exp_id}] Skipping — already done")
            with open(final_path) as f:
                all_results.append(json.load(f))
            continue

        alpha = param if aggregation == "fedavg" else None
        mu    = param if aggregation == "fedprox" else 0.0

        try:
            result = run_experiment(
                experiment_id = exp_id,
                model_key     = model_key,
                aggregation   = aggregation,
                non_iid_type  = non_iid_type,
                alpha         = alpha,
                mu            = mu,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"[{exp_id}] FAILED: {e}")
            logger.error(traceback.format_exc())
            failed.append(exp_id)

    # Summary
    if all_results:
        summary = pd.DataFrame(all_results)
        path = os.path.join(config.RESULTS_DIR, "all_experiments_summary.csv")
        summary.to_csv(path, index=False)
        logger.info(f"\nSummary saved to {path}")
        print("\n" + "="*70)
        print(summary[["experiment_id","model","aggregation","non_iid_type","accuracy","f1","convergence_round"]].to_string(index=False))
        print("="*70)

    if failed:
        logger.warning(f"Failed: {failed}")


if __name__ == "__main__":
    main()
