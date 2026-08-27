from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.partition import get_partitions
from data.preprocess import (
    compute_vocabulary_overlap,
    load_and_merge,
    split_dataset,
)
from models.transformer_classifier import (
    DEVICE,
    EmailDataset,
    evaluate,
    get_model,
    get_parameters,
    local_train,
    set_global_seed,
    set_parameters,
)
 
os.makedirs(config.RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(config.RESULTS_DIR, "sequential_run.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


def fedavg_aggregate(
    all_parameters: list[list[np.ndarray]],
    all_sizes: list[int],
) -> list[np.ndarray]:
    """Compute sample-size-weighted FedAvg without changing buffer dtypes."""
    if not all_parameters:
        raise ValueError("No client parameters were provided for aggregation.")
    if len(all_parameters) != len(all_sizes):
        raise ValueError("Parameter and client-size lists must have equal length.")
    if any(size <= 0 for size in all_sizes):
        raise ValueError("All participating client sizes must be positive.")

    layer_count = len(all_parameters[0])
    if any(len(parameters) != layer_count for parameters in all_parameters):
        raise ValueError("Clients returned different numbers of model tensors.")

    total_examples = float(sum(all_sizes))
    weights = [size / total_examples for size in all_sizes]
    aggregated: list[np.ndarray] = []

    for layer_index in range(layer_count):
        reference = all_parameters[0][layer_index]
        for client_parameters in all_parameters[1:]:
            if client_parameters[layer_index].shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch at state tensor {layer_index}: "
                    f"{client_parameters[layer_index].shape} != {reference.shape}."
                )

        if np.issubdtype(reference.dtype, np.floating) or np.issubdtype(
            reference.dtype, np.complexfloating
        ):
            averaged = np.zeros_like(reference)
            for client_parameters, weight in zip(all_parameters, weights):
                averaged += client_parameters[layer_index] * weight
            aggregated.append(averaged)
        else:
            # Integer/bool buffers are not trainable and should be identical.
            aggregated.append(reference.copy())

    return aggregated


def _weighted_mean(values: list[float], sizes: list[int]) -> float:
    total = sum(sizes)
    return sum(value * size for value, size in zip(values, sizes)) / total


def _copy_parameters(parameters: list[np.ndarray]) -> list[np.ndarray]:
    return [array.copy() for array in parameters]


def _partition_summary(partitions: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for client_id, partition in enumerate(partitions, start=1):
        positives = int(partition[config.LABEL_COLUMN].sum())
        total = len(partition)
        rows.append(
            {
                "client": client_id,
                "total": total,
                "phishing_or_spam": positives,
                "legitimate": total - positives,
                "positive_fraction": positives / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    experiment_id: str,
    model_key: str,
    aggregation: str,
    non_iid_type: str,
    alpha: float | None = None,
    mu: float = 0.0,
    num_rounds: int = config.NUM_ROUNDS,
    num_clients: int = config.NUM_CLIENTS,
) -> dict:
    """Run one complete federated experiment and save round/final results."""
    if aggregation not in {"fedavg", "fedprox"}:
        raise ValueError("aggregation must be 'fedavg' or 'fedprox'.")
    if aggregation == "fedavg" and mu != 0.0:
        raise ValueError("FedAvg must use mu=0.0.")
    if aggregation == "fedprox" and mu <= 0.0:
        raise ValueError("FedProx must use a positive mu.")

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

    # Reset all random generators so paired FedAvg/FedProx experiments begin
    # from the same model head and deterministic local minibatch orders.
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

    if len(partitions) != num_clients:
        raise RuntimeError(
            f"Expected {num_clients} client partitions, got {len(partitions)}."
        )
    if any(len(partition) == 0 for partition in partitions):
        raise RuntimeError("All five clients must remain active; an empty partition exists.")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    partition_path = os.path.join(
        config.RESULTS_DIR, f"{experiment_id}_partition_summary.csv"
    )
    _partition_summary(partitions).to_csv(partition_path, index=False)

    if non_iid_type == "feature_skew":
        overlap = compute_vocabulary_overlap(
            [partition[config.TEXT_COLUMN].tolist() for partition in partitions]
        )
        overlap.to_csv(
            os.path.join(config.RESULTS_DIR, f"{experiment_id}_vocab_overlap.csv")
        )

    # Reset immediately before model construction. This guarantees identical
    # initial classifier weights for paired experiments using the same model.
    set_global_seed(config.SEED)
    model, tokenizer = get_model(model_key)

    client_datasets = [
        EmailDataset(
            {
                config.TEXT_COLUMN: partition[config.TEXT_COLUMN].tolist(),
                config.LABEL_COLUMN: partition[config.LABEL_COLUMN].tolist(),
            },
            tokenizer,
        )
        for partition in partitions
    ]
    val_dataset = EmailDataset(
        {
            config.TEXT_COLUMN: val_df[config.TEXT_COLUMN].tolist(),
            config.LABEL_COLUMN: val_df[config.LABEL_COLUMN].tolist(),
        },
        tokenizer,
    )
    test_dataset = EmailDataset(
        {
            config.TEXT_COLUMN: test_df[config.TEXT_COLUMN].tolist(),
            config.LABEL_COLUMN: test_df[config.LABEL_COLUMN].tolist(),
        },
        tokenizer,
    )

    global_parameters = get_parameters(model)
    best_parameters = _copy_parameters(global_parameters)
    round_results: list[dict] = []

    best_f1 = float("-inf")
    best_validation_round = 0
    no_improve = 0

    for server_round in range(1, num_rounds + 1):
        round_start = time.time()
        logger.info("[Round %d/%d]", server_round, num_rounds)

        client_parameters: list[list[np.ndarray]] = []
        client_sizes: list[int] = []
        classification_losses: list[float] = []
        objective_losses: list[float] = []

        for client_id, client_dataset in enumerate(client_datasets):
            logger.info(
                "  Client %d/%d training | examples=%d",
                client_id + 1,
                num_clients,
                len(client_dataset),
            )
            set_parameters(model, global_parameters)

            local_seed = config.SEED + server_round * 10_000 + client_id
            _, num_examples, losses = local_train(
                model=model,
                train_dataset=client_dataset,
                num_epochs=config.LOCAL_EPOCHS,
                batch_size=config.LOCAL_BATCH,
                lr=config.LEARNING_RATE,
                mu=mu,
                seed=local_seed,
            )

            client_parameters.append(get_parameters(model))
            client_sizes.append(num_examples)
            classification_losses.append(losses["classification_loss"])
            objective_losses.append(losses["objective_loss"])

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        global_parameters = fedavg_aggregate(client_parameters, client_sizes)
        set_parameters(model, global_parameters)
        validation_metrics = evaluate(model, val_dataset, batch_size=32)

        record = {
            "experiment_id": experiment_id,
            "round": server_round,
            "loss": validation_metrics["loss"],
            "accuracy": validation_metrics["accuracy"],
            "f1": validation_metrics["f1"],
            "precision": validation_metrics["precision"],
            "recall": validation_metrics["recall"],
            "auc_roc": validation_metrics["auc_roc"],
            "classification_loss": _weighted_mean(
                classification_losses, client_sizes
            ),
            "objective_loss": _weighted_mean(objective_losses, client_sizes),
            "round_time_sec": round(time.time() - round_start, 1),
        }
        round_results.append(record)

        logger.info(
            "[Round %d] %s | accuracy=%.4f | macro_f1=%.4f | auc=%.4f "
            "| validation_loss=%.4f | time=%.1fs",
            server_round,
            aggregation.upper(),
            validation_metrics["accuracy"],
            validation_metrics["f1"],
            validation_metrics["auc_roc"],
            validation_metrics["loss"],
            record["round_time_sec"],
        )

        if validation_metrics["f1"] > best_f1 + config.EARLY_STOP_DELTA:
            best_f1 = float(validation_metrics["f1"])
            best_validation_round = server_round
            best_parameters = _copy_parameters(global_parameters)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_ROUNDS:
                logger.info(
                    "Early stopping at round %d; best validation round=%d.",
                    server_round,
                    best_validation_round,
                )
                break

    if best_validation_round == 0:
        raise RuntimeError("No validation checkpoint was recorded.")

    # Test only once, using the checkpoint selected exclusively by validation F1.
    set_parameters(model, best_parameters)
    test_metrics = evaluate(model, test_dataset, batch_size=32)

    total_rounds = len(round_results)
    final = {
        "code_version": config.CODE_VERSION,
        "experiment_id": experiment_id,
        "model": model_key,
        "aggregation": aggregation,
        "non_iid_type": non_iid_type,
        "alpha": alpha,
        "mu": mu,
        "accuracy": test_metrics["accuracy"],
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "auc_roc": test_metrics["auc_roc"],
        "loss": test_metrics["loss"],
        # Retained for compatibility with the paper's Conv./Total Rnds column.
        "convergence_round": best_validation_round,
        "best_validation_round": best_validation_round,
        "early_stopping_round": total_rounds,
        "total_rounds": total_rounds,
        "runtime_sec": round(time.time() - start_time, 1),
        "train_examples": len(train_df),
        "validation_examples": len(val_df),
        "test_examples": len(test_df),
        "num_clients": num_clients,
        "seed": config.SEED,
    }

    rounds_path = os.path.join(config.RESULTS_DIR, f"{experiment_id}_rounds.csv")
    pd.DataFrame(round_results).to_csv(rounds_path, index=False)

    final_path = os.path.join(config.RESULTS_DIR, f"{experiment_id}_final.json")
    with open(final_path, "w", encoding="utf-8") as output_file:
        json.dump(final, output_file, indent=2)

    logger.info(
        "[%s] complete | test_accuracy=%.4f | test_macro_f1=%.4f "
        "| best_round=%d | total_rounds=%d",
        experiment_id,
        final["accuracy"],
        final["f1"],
        best_validation_round,
        total_rounds,
    )

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return final


def _matrix_parameters(
    aggregation: str,
    parameter: float | None,
) -> tuple[float | None, float]:
    if aggregation == "fedavg":
        return parameter, 0.0
    return None, float(parameter if parameter is not None else config.FEDPROX_MU_PRIMARY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected E1-E20 experiments.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    matrix = config.EXPERIMENT_MATRIX
    if len(matrix) != 20 or any(not isinstance(row, tuple) or len(row) != 5 for row in matrix):
        raise RuntimeError("EXPERIMENT_MATRIX must contain exactly 20 five-field tuples.")

    if args.only:
        requested = set(args.only)
        known = {row[0] for row in matrix}
        unknown = requested.difference(known)
        if unknown:
            raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
        matrix = [row for row in matrix if row[0] in requested]

    all_results: list[dict] = []
    failed: list[str] = []

    for experiment_id, model_key, aggregation, non_iid_type, parameter in matrix:
        final_path = os.path.join(
            config.RESULTS_DIR, f"{experiment_id}_final.json"
        )

        if args.skip_existing and os.path.exists(final_path):
            with open(final_path, encoding="utf-8") as existing_file:
                existing = json.load(existing_file)
            if existing.get("code_version") == config.CODE_VERSION:
                logger.info("[%s] skipping compatible existing result.", experiment_id)
                all_results.append(existing)
                continue
            logger.info(
                "[%s] existing result uses an older code version; rerunning.",
                experiment_id,
            )

        alpha, mu = _matrix_parameters(aggregation, parameter)
        try:
            result = run_experiment(
                experiment_id=experiment_id,
                model_key=model_key,
                aggregation=aggregation,
                non_iid_type=non_iid_type,
                alpha=alpha,
                mu=mu,
            )
            all_results.append(result)
        except Exception as exc:  # continue remaining experiments, preserve traceback
            logger.error("[%s] failed: %s", experiment_id, exc)
            logger.error(traceback.format_exc())
            failed.append(experiment_id)

    if all_results:
        summary = pd.DataFrame(all_results).sort_values("experiment_id")
        summary_path = os.path.join(
            config.RESULTS_DIR, "all_experiments_summary.csv"
        )
        summary.to_csv(summary_path, index=False)
        display_columns = [
            "experiment_id",
            "model",
            "aggregation",
            "non_iid_type",
            "accuracy",
            "f1",
            "convergence_round",
            "total_rounds",
        ]
        print("\n" + "=" * 100)
        print(summary[display_columns].to_string(index=False))
        print("=" * 100)
        logger.info("Summary saved to %s", summary_path)

    if failed:
        logger.warning("Failed experiments: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
