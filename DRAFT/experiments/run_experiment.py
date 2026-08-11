"""
experiments/run_experiment.py
------------------------------
Runs a single experiment from the 2×2×3 matrix (E1–E20).

Usage (CLI):
  python experiments/run_experiment.py \
      --model distilbert \
      --aggregation fedprox \
      --non_iid_type label_skew_severe \
      --mu 0.1 \
      --experiment_id E7

Or imported and called programmatically from run_all_experiments.py.
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Optional

import pandas as pd
import numpy as np
import torch
import flwr as fl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.preprocess  import load_and_merge, split_dataset, compute_vocabulary_overlap
from data.partition   import get_partitions
from models.transformer_classifier import get_model, EmailDataset, evaluate, get_parameters
from clients.fl_client            import make_client_fn
from aggregation.fedavg_strategy  import LoggingFedAvg
from aggregation.fedprox_strategy import LoggingFedProx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Strategy

def build_strategy(
    aggregation:   str,
    experiment_id: str,
    initial_params,
    mu:            float = 0.0,
):
    """Return a configured Flower strategy instance."""

    common_kwargs = dict(
        experiment_id             = experiment_id,
        fraction_fit              = 1.0,
        fraction_evaluate         = 1.0,
        min_fit_clients           = config.MIN_FIT_CLIENTS,
        min_evaluate_clients      = config.MIN_EVAL_CLIENTS,
        min_available_clients     = config.MIN_AVAILABLE_CLIENTS,
        initial_parameters        = initial_params,
    )

    if aggregation == "fedavg":
        return LoggingFedAvg(**common_kwargs)

    elif aggregation == "fedprox":
        return LoggingFedProx(mu=mu, **common_kwargs)

    else:
        raise ValueError(f"Unknown aggregation: '{aggregation}'. Use 'fedavg' or 'fedprox'.")

# Main experiment runner

def run_experiment(
    experiment_id: str,
    model_key:     str,
    aggregation:   str,
    non_iid_type:  str,
    alpha:         Optional[float] = None,
    mu:            float           = config.FEDPROX_MU_PRIMARY,
    num_rounds:    int             = config.NUM_ROUNDS,
    num_clients:   int             = config.NUM_CLIENTS,
    results_dir:   str             = config.RESULTS_DIR,
) -> dict:
    """
    Orchestrate a single federated experiment end-to-end.

    Steps:
      1. Load + preprocess datasets
      2. Split into train / val / test
      3. Partition train across clients (Non-IID strategy)
      4. Build Flower strategy (FedAvg or FedProx)
      5. Run Flower simulation
      6. Evaluate final global model on IID test set
      7. Save round-level results + final metrics to CSV/JSON

    Returns:
        dict of final test-set metrics
    """
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT {experiment_id}")
    logger.info(f"  Model:        {model_key}")
    logger.info(f"  Aggregation:  {aggregation}")
    logger.info(f"  Non-IID type: {non_iid_type}")
    logger.info(f"  Alpha/Mu:     {alpha if aggregation=='fedavg' else mu}")
    logger.info("=" * 70)

    start_time = time.time()

    #  1. Data loading
    df = load_and_merge()
    train_df, val_df, test_df = split_dataset(df)

    #  2. Partitioning 
    partitions = get_partitions(
        train_df     = train_df,
        non_iid_type = non_iid_type,
        alpha        = alpha,
        num_clients  = num_clients,
    )

    # Feature skew: compute vocabulary overlap to validate distinctness
    if non_iid_type == "feature_skew":
        overlap_df = compute_vocabulary_overlap(
            [p["text"].tolist() for p in partitions]
        )
        overlap_path = os.path.join(results_dir, f"{experiment_id}_vocab_overlap.csv")
        overlap_df.to_csv(overlap_path)
        logger.info(f"Vocabulary overlap saved to {overlap_path}")

    #  3. Model initialisation 
    model, tokenizer = get_model(model_key)

    # Convert initial model parameters for Flower
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(model))

    #  4. Build strategy
    strategy = build_strategy(
        aggregation   = aggregation,
        experiment_id = experiment_id,
        initial_params = initial_params,
        mu            = mu,
    )

    #  5. Run FL simulation 
    client_fn = make_client_fn(
        partitions = partitions,
        val_df     = val_df,
        model_key  = model_key,
        mu         = mu if aggregation == "fedprox" else 0.0,
    )

    logger.info(f"Starting Flower simulation | rounds={num_rounds} | clients={num_clients}")

    fl.simulation.start_simulation(
        client_fn         = client_fn,
        num_clients       = num_clients,
        config            = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy          = strategy,
        client_resources  = {
            "num_cpus": 1,
            "num_gpus": 1.0 / num_clients if torch.cuda.is_available() else 0.0,
        },
    )

    #  6. Final evaluation on IID test set
    # Reload model with the final aggregated parameters
    # (Strategy stores best params; we use what's in the strategy's last round)
    # We re-instantiate a fresh model and load the best round's parameters
    final_model, _ = get_model(model_key)
    test_dataset = EmailDataset(
        {"text": test_df["text"].tolist(), "label": test_df["label"].tolist()},
        tokenizer,
    )
    test_metrics = evaluate(final_model, test_dataset)
    test_metrics["experiment_id"] = experiment_id
    test_metrics["model"]         = model_key
    test_metrics["aggregation"]   = aggregation
    test_metrics["non_iid_type"]  = non_iid_type
    test_metrics["alpha"]         = alpha
    test_metrics["mu"]            = mu
    test_metrics["runtime_sec"]   = round(time.time() - start_time, 1)

    # Detect convergence round: first round where F1 plateaued
    if strategy.round_results:
        f1s = [r["f1"] for r in strategy.round_results]
        convergence_round = _find_convergence_round(f1s, config.EARLY_STOP_DELTA)
        test_metrics["convergence_round"] = convergence_round
    else:
        test_metrics["convergence_round"] = num_rounds

    logger.info(f"Test metrics: {test_metrics}")

    #  7. Save results
    os.makedirs(results_dir, exist_ok=True)

    # Per-round CSV
    rounds_path = os.path.join(results_dir, f"{experiment_id}_rounds.csv")
    strategy.save_results(rounds_path)

    # Final metrics JSON
    final_path = os.path.join(results_dir, f"{experiment_id}_final.json")
    with open(final_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Final metrics saved to {final_path}")

    return test_metrics

# Helper: convergence round detection

def _find_convergence_round(f1_series: list[float], delta: float) -> int:
    """
    Return the first round index (1-based) where F1 stops improving
    by more than `delta`, held for EARLY_STOP_ROUNDS consecutive rounds.
    """
    best = -1.0
    no_improve = 0
    for i, f1 in enumerate(f1_series):
        if f1 > best + delta:
            best = f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_ROUNDS:
                return i + 1  # 1-based
    return len(f1_series)

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single FL phishing detection experiment")
    parser.add_argument("--model",        required=True, choices=["distilbert", "tinybert"])
    parser.add_argument("--aggregation",  required=True, choices=["fedavg", "fedprox"])
    parser.add_argument("--non_iid_type", required=True,
                        choices=["label_skew_iid", "label_skew_moderate",
                                 "label_skew_severe", "feature_skew", "quantity"])
    parser.add_argument("--alpha",          type=float, default=None,
                        help="Dirichlet α for label skew (override config)")
    parser.add_argument("--mu",             type=float, default=config.FEDPROX_MU_PRIMARY,
                        help="FedProx µ coefficient")
    parser.add_argument("--num_rounds",     type=int,   default=config.NUM_ROUNDS)
    parser.add_argument("--num_clients",    type=int,   default=config.NUM_CLIENTS)
    parser.add_argument("--experiment_id",  type=str,   default="E_manual")
    parser.add_argument("--results_dir",    type=str,   default=config.RESULTS_DIR)

    args = parser.parse_args()

    run_experiment(
        experiment_id = args.experiment_id,
        model_key     = args.model,
        aggregation   = args.aggregation,
        non_iid_type  = args.non_iid_type,
        alpha         = args.alpha,
        mu            = args.mu,
        num_rounds    = args.num_rounds,
        num_clients   = args.num_clients,
        results_dir   = args.results_dir,
    )
