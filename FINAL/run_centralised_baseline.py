from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.preprocess import load_and_merge, split_dataset
from models.transformer_classifier import (
    DEVICE,
    EmailDataset,
    evaluate,
    get_model,
    get_parameters,
    set_global_seed,
    set_parameters,
    train_one_epoch,
)

BASELINE_IDS = {
    "distilbert": "CB-D",
    "tinybert": "CB-T",
}

CENTRALISED_CODE_VERSION = f"{config.CODE_VERSION}-centralised-v1"
CENTRALISED_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "centralised")
os.makedirs(CENTRALISED_RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(CENTRALISED_RESULTS_DIR, "centralised_run.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


def _copy_parameters(parameters: list[Any]) -> list[Any]:
    """Return independent CPU copies of model-state arrays."""
    return [parameter.copy() for parameter in parameters]


def _make_dataset(frame: pd.DataFrame, tokenizer: Any) -> EmailDataset:
    return EmailDataset(
        {
            config.TEXT_COLUMN: frame[config.TEXT_COLUMN].tolist(),
            config.LABEL_COLUMN: frame[config.LABEL_COLUMN].tolist(),
        },
        tokenizer,
    )


def run_centralised_baseline(
    model_key: str,
    max_epochs: int = config.NUM_ROUNDS,
    batch_size: int = config.LOCAL_BATCH,
    learning_rate: float = config.LEARNING_RATE,
) -> dict[str, Any]:
    """Train one model on the complete training split and test its best checkpoint."""
    if model_key not in BASELINE_IDS:
        raise ValueError(
            f"Unknown model {model_key!r}; choose from {sorted(BASELINE_IDS)}."
        )
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    baseline_id = BASELINE_IDS[model_key]

    logger.info("=" * 72)
    logger.info(
        "BASELINE %s | model=%s | centralised full-training split",
        baseline_id,
        model_key,
    )
    logger.info("=" * 72)

    start_time = time.time()
    set_global_seed(config.SEED)

    dataset = load_and_merge(seed=config.SEED)
    train_df, val_df, test_df = split_dataset(dataset, seed=config.SEED)

    # Reset directly before model construction so the classification head is
    # initialised deterministically, matching the federated experiment setup.
    set_global_seed(config.SEED)
    model, tokenizer = get_model(model_key)

    train_dataset = _make_dataset(train_df, tokenizer)
    val_dataset = _make_dataset(val_df, tokenizer)
    test_dataset = _make_dataset(test_df, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=DEVICE.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = max(len(train_loader) * max_epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    initial_parameters = get_parameters(model)
    best_parameters = _copy_parameters(initial_parameters)
    best_validation_f1 = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    epoch_results: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        classification_loss, objective_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            mu=0.0,
            global_params=None,
        )

        validation_metrics = evaluate(model, val_dataset, batch_size=32)

        record = {
            "baseline_id": baseline_id,
            "model": model_key,
            "epoch": epoch,
            "classification_loss": classification_loss,
            "objective_loss": objective_loss,
            "validation_loss": validation_metrics["loss"],
            "accuracy": validation_metrics["accuracy"],
            "f1": validation_metrics["f1"],
            "precision": validation_metrics["precision"],
            "recall": validation_metrics["recall"],
            "auc_roc": validation_metrics["auc_roc"],
            "epoch_time_sec": round(time.time() - epoch_start, 1),
        }
        epoch_results.append(record)

        logger.info(
            "[Epoch %d/%d] accuracy=%.4f | macro_f1=%.4f | auc=%.4f "
            "| validation_loss=%.4f | time=%.1fs",
            epoch,
            max_epochs,
            validation_metrics["accuracy"],
            validation_metrics["f1"],
            validation_metrics["auc_roc"],
            validation_metrics["loss"],
            record["epoch_time_sec"],
        )

        # Use the same macro-F1 improvement rule as run_sequential.py so
        # checkpoint selection and early stopping are directly comparable.
        if (
            validation_metrics["f1"]
            > best_validation_f1 + config.EARLY_STOP_DELTA
        ):
            best_validation_f1 = validation_metrics["f1"]
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_parameters = _copy_parameters(get_parameters(model))
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.EARLY_STOP_ROUNDS:
            logger.info(
                "Early stopping at epoch %d; best validation epoch=%d.",
                epoch,
                best_epoch,
            )
            break

    if best_epoch == 0:
        raise RuntimeError("No centralised checkpoint was selected.")

    # Evaluate only once on the untouched test set, after model selection using
    # validation data.
    set_parameters(model, best_parameters)
    test_metrics = evaluate(model, test_dataset, batch_size=32)

    final = {
        "code_version": CENTRALISED_CODE_VERSION,
        "baseline_id": baseline_id,
        "training_mode": "centralised",
        "model": model_key,
        "seed": config.SEED,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": config.WEIGHT_DECAY,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "test_examples": len(test_dataset),
        "accuracy": test_metrics["accuracy"],
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "auc_roc": test_metrics["auc_roc"],
        "loss": test_metrics["loss"],
        "best_validation_f1": best_validation_f1,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "convergence_epoch": best_epoch,
        "total_epochs": len(epoch_results),
        "runtime_sec": round(time.time() - start_time, 1),
    }

    rounds_path = os.path.join(
        CENTRALISED_RESULTS_DIR, f"{baseline_id}_epochs.csv"
    )
    final_path = os.path.join(
        CENTRALISED_RESULTS_DIR, f"{baseline_id}_final.json"
    )

    pd.DataFrame(epoch_results).to_csv(rounds_path, index=False)
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)

    logger.info(
        "[%s] complete | test_accuracy=%.4f | test_macro_f1=%.4f "
        "| best_epoch=%d | total_epochs=%d",
        baseline_id,
        final["accuracy"],
        final["f1"],
        final["best_epoch"],
        final["total_epochs"],
    )

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run centralised DistilBERT and TinyBERT baselines."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(BASELINE_IDS),
        default=None,
        help="Run one or both models. Default: both.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse a result only when its code version matches this runner.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=config.NUM_ROUNDS,
        help="Maximum centralised epochs; default matches NUM_ROUNDS.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.LOCAL_BATCH,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.LEARNING_RATE,
    )
    args = parser.parse_args()

    selected_models = args.only or ["distilbert", "tinybert"]
    completed: list[dict[str, Any]] = []
    failed: list[str] = []

    for model_key in selected_models:
        baseline_id = BASELINE_IDS[model_key]
        final_path = os.path.join(
            CENTRALISED_RESULTS_DIR, f"{baseline_id}_final.json"
        )

        if args.skip_existing and os.path.exists(final_path):
            with open(final_path, encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("code_version") == CENTRALISED_CODE_VERSION:
                logger.info("[%s] skipping compatible existing result.", baseline_id)
                completed.append(existing)
                continue
            logger.info(
                "[%s] existing result has a different code version; rerunning.",
                baseline_id,
            )

        try:
            completed.append(
                run_centralised_baseline(
                    model_key=model_key,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
            )
        except Exception as exc:
            failed.append(baseline_id)
            logger.error("[%s] FAILED: %s", baseline_id, exc)
            logger.error(traceback.format_exc())

    if completed:
        summary = pd.DataFrame(completed).sort_values("baseline_id")
        summary_path = os.path.join(
            CENTRALISED_RESULTS_DIR,
            "centralised_baselines_summary.csv",
        )
        summary.to_csv(summary_path, index=False)

        display_columns = [
            "baseline_id",
            "model",
            "accuracy",
            "f1",
            "best_epoch",
            "total_epochs",
        ]
        print("\n" + "=" * 88)
        print(summary[display_columns].to_string(index=False))
        print("=" * 88)
        logger.info("Summary saved to %s", summary_path)

    if failed:
        logger.warning("Failed baselines: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
