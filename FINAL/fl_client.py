from __future__ import annotations

import logging
import os
import sys
from typing import Any

import flwr as fl
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.transformer_classifier import (
    EmailDataset,
    evaluate,
    get_parameters,
    local_train,
    set_parameters,
)

logger = logging.getLogger(__name__)


class PhishingClient(fl.client.NumPyClient):
    """A single simulated organisation participating in FL training."""

    def __init__(
        self,
        client_id: int,
        model: Any,
        tokenizer: Any,
        train_df: Any,
        val_df: Any,
        mu: float = 0.0,
    ) -> None:
        if len(train_df) == 0:
            raise ValueError(f"Client {client_id} received an empty training partition.")

        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.mu = mu

        self.train_dataset = EmailDataset(
            {
                config.TEXT_COLUMN: train_df[config.TEXT_COLUMN].tolist(),
                config.LABEL_COLUMN: train_df[config.LABEL_COLUMN].tolist(),
            },
            tokenizer,
        )
        self.val_dataset = EmailDataset(
            {
                config.TEXT_COLUMN: val_df[config.TEXT_COLUMN].tolist(),
                config.LABEL_COLUMN: val_df[config.LABEL_COLUMN].tolist(),
            },
            tokenizer,
        )

        logger.info(
            "Client %d initialised | train=%d | validation=%d | mu=%s",
            client_id,
            len(self.train_dataset),
            len(self.val_dataset),
            mu,
        )

    def get_parameters(self, config_dict: dict) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: list[np.ndarray],
        fit_config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        set_parameters(self.model, parameters)

        mu = float(fit_config.get("mu", self.mu))
        server_round = int(fit_config.get("round", 0))
        local_seed = config.SEED + server_round * 10_000 + self.client_id

        _, num_examples, losses = local_train(
            model=self.model,
            train_dataset=self.train_dataset,
            num_epochs=int(fit_config.get("local_epochs", config.LOCAL_EPOCHS)),
            batch_size=int(fit_config.get("batch_size", config.LOCAL_BATCH)),
            lr=float(fit_config.get("lr", config.LEARNING_RATE)),
            mu=mu,
            seed=local_seed,
        )

        logger.info(
            "Client %d | fit complete | examples=%d | classification_loss=%.4f "
            "| objective_loss=%.4f",
            self.client_id,
            num_examples,
            losses["classification_loss"],
            losses["objective_loss"],
        )

        return get_parameters(self.model), num_examples, losses

    def evaluate(
        self,
        parameters: list[np.ndarray],
        eval_config: dict,
    ) -> tuple[float, int, dict]:
        set_parameters(self.model, parameters)
        metrics = evaluate(self.model, self.val_dataset)

        logger.info(
            "Client %d | validation | accuracy=%.4f | macro_f1=%.4f",
            self.client_id,
            metrics["accuracy"],
            metrics["f1"],
        )

        return (
            float(metrics["loss"]),
            int(metrics["num_examples"]),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "auc_roc": float(metrics["auc_roc"]),
            },
        )


def make_client_fn(
    partitions: list,
    val_df: Any,
    model_key: str,
    mu: float = 0.0,
):
    """Create the client factory required by Flower simulation."""
    from models.transformer_classifier import get_model, set_global_seed

    client_models: dict[int, tuple[Any, Any]] = {}

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        if client_id < 0 or client_id >= len(partitions):
            raise IndexError(f"Invalid client id {client_id}.")

        if client_id not in client_models:
            # Each client model has the same initial sequence-classification head.
            set_global_seed(config.SEED)
            client_models[client_id] = get_model(model_key)

        model, tokenizer = client_models[client_id]
        return PhishingClient(
            client_id=client_id,
            model=model,
            tokenizer=tokenizer,
            train_df=partitions[client_id],
            val_df=val_df,
            mu=mu,
        ).to_client()

    return client_fn
