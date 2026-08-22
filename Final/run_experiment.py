"""Flower-based runner for one experiment from E1-E20.

The sequential runner in the project root is recommended for a 4 GB GPU. This
module remains available for Flower/Ray environments and now loads the selected
best aggregated checkpoint before final IID test evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from aggregation.fedavg_strategy import LoggingFedAvg
from aggregation.fedprox_strategy import LoggingFedProx
from clients.fl_client import make_client_fn
from data.partition import get_partitions
from data.preprocess import (
    compute_vocabulary_overlap,
    load_and_merge,
    split_dataset,
)
from models.transformer_classifier import (
    EmailDataset,
    evaluate,
    get_model,
    get_parameters,
    set_global_seed,
    set_parameters,
)

logger = logging.getLogger(__name__)


def build_strategy(
    aggregation: str,
    experiment_id: str,
    initial_parameters,
    mu: float = 0.0,
):
    """Create the selected Flower strategy."""
    common_kwargs = {
        "experiment_id": experiment_id,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": config.MIN_FIT_CLIENTS,
        "min_evaluate_clients": config.MIN_EVAL_CLIENTS,
        "min_available_clients": config.MIN_AVAILABLE_CLIENTS,
        "initial_parameters": initial_parameters,
    }

    if aggregation == "fedavg":
        if mu != 0.0:
            raise ValueError("FedAvg must use mu=0.0.")
        return LoggingFedAvg(**common_kwargs)
    if aggregation == "fedprox":
        if mu <= 0.0:
            raise ValueError("FedProx must use a positive mu.")
        return LoggingFedProx(mu=mu, **common_kwargs)
    raise ValueError("aggregation must be 'fedavg' or 'fedprox'.")


def run_experiment(
    experiment_id: str,
    model_key: str,
    aggregation: str,
    non_iid_type: str,
    alpha: float | None = None,
    mu: float = config.FEDPROX_MU_PRIMARY,
    num_rounds: int = config.NUM_ROUNDS,
    num_clients: int = config.NUM_CLIENTS,
    results_dir: str = config.RESULTS_DIR,
) -> dict:
    """Run one Flower simulation and test its best validation checkpoint."""
    logger.info("=" * 70)
    logger.info(
        "EXPERIMENT %s | model=%s | aggregation=%s | condition=%s",
        experiment_id,
        model_key,
        aggregation,
        non_iid_type,
    )
    logger.info("=" * 70)
    start_time = time.time()
    os.makedirs(results_dir, exist_ok=True)

    set_global_seed(config.SEED)
    dataset = load_and_merge(seed=config.SEED)
    train_df, val_df, test_df = split_dataset(dataset, seed=config.SEED)
    partitions = get_partitions(
        train_df=train_df,
        non_iid_type=non_iid_type,
        alpha=alpha,
        num_clients=num_clients,
        seed=config.SEED,
    )
    if any(len(partition) == 0 for partition in partitions):
        raise RuntimeError("All configured clients must have non-empty partitions.")

    if non_iid_type == "feature_skew":
        overlap = compute_vocabulary_overlap(
            [partition[config.TEXT_COLUMN].tolist() for partition in partitions]
        )
        overlap.to_csv(
            os.path.join(results_dir, f"{experiment_id}_vocab_overlap.csv")
        )

    set_global_seed(config.SEED)
    initial_model, tokenizer = get_model(model_key)
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_parameters(initial_model)
    )

    strategy = build_strategy(
        aggregation=aggregation,
        experiment_id=experiment_id,
        initial_parameters=initial_parameters,
        mu=mu if aggregation == "fedprox" else 0.0,
    )
    client_fn = make_client_fn(
        partitions=partitions,
        val_df=val_df,
        model_key=model_key,
        mu=mu if aggregation == "fedprox" else 0.0,
    )

    logger.info(
        "Starting Flower simulation | rounds=%d | clients=%d",
        num_rounds,
        num_clients,
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 1.0 / num_clients if torch.cuda.is_available() else 0.0,
        },
    )

    # Critical correction: load the selected aggregated checkpoint rather than
    # evaluating a fresh pretrained model with a random classification head.
    set_global_seed(config.SEED)
    final_model, final_tokenizer = get_model(model_key)
    selected_parameters = parameters_to_ndarrays(strategy.parameters_for_testing())
    set_parameters(final_model, selected_parameters)

    test_dataset = EmailDataset(
        {
            config.TEXT_COLUMN: test_df[config.TEXT_COLUMN].tolist(),
            config.LABEL_COLUMN: test_df[config.LABEL_COLUMN].tolist(),
        },
        final_tokenizer,
    )
    test_metrics = evaluate(final_model, test_dataset)

    total_rounds = len(strategy.round_results)
    final = {
        "code_version": config.CODE_VERSION,
        "experiment_id": experiment_id,
        "model": model_key,
        "aggregation": aggregation,
        "non_iid_type": non_iid_type,
        "alpha": alpha,
        "mu": mu if aggregation == "fedprox" else 0.0,
        "accuracy": test_metrics["accuracy"],
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "auc_roc": test_metrics["auc_roc"],
        "loss": test_metrics["loss"],
        "convergence_round": strategy.best_round,
        "best_validation_round": strategy.best_round,
        "plateau_detected_round": strategy.plateau_detected_round,
        "total_rounds": total_rounds,
        "runtime_sec": round(time.time() - start_time, 1),
        "seed": config.SEED,
        "num_clients": num_clients,
    }

    strategy.save_results(
        os.path.join(results_dir, f"{experiment_id}_rounds.csv")
    )
    final_path = os.path.join(results_dir, f"{experiment_id}_final.json")
    with open(final_path, "w", encoding="utf-8") as output_file:
        json.dump(final, output_file, indent=2)

    logger.info("Final test metrics: %s", final)
    return final


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run one corrected E1-E20 Flower experiment."
    )
    parser.add_argument(
        "--model", required=True, choices=["distilbert", "tinybert"]
    )
    parser.add_argument(
        "--aggregation", required=True, choices=["fedavg", "fedprox"]
    )
    parser.add_argument(
        "--non_iid_type",
        required=True,
        choices=sorted(config.VALID_NON_IID_TYPES),
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--mu", type=float, default=config.FEDPROX_MU_PRIMARY)
    parser.add_argument("--num_rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--num_clients", type=int, default=config.NUM_CLIENTS)
    parser.add_argument("--experiment_id", type=str, default="E_manual")
    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR)
    arguments = parser.parse_args()

    run_experiment(
        experiment_id=arguments.experiment_id,
        model_key=arguments.model,
        aggregation=arguments.aggregation,
        non_iid_type=arguments.non_iid_type,
        alpha=arguments.alpha,
        mu=arguments.mu,
        num_rounds=arguments.num_rounds,
        num_clients=arguments.num_clients,
        results_dir=arguments.results_dir,
    )
