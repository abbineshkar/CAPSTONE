"""
clients/fl_client.py
---------------------
Flower (flwr) client implementation.

Each client:
  1. Receives global model parameters from the server
  2. Loads its local partition into an EmailDataset
  3. Runs local_train() for LOCAL_EPOCHS epochs (with optional FedProx µ)
  4. Returns updated parameters + metrics to the server

Supports both FedAvg (µ=0) and FedProx (µ>0) transparently.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.transformer_classifier import (
    EmailDataset,
    get_parameters,
    set_parameters,
    local_train,
    evaluate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PhishingClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for federated phishing email classification.

    Args:
        client_id:     Integer ID for this client (0-indexed)
        model:         Pre-loaded transformer model (distilbert / tinybert)
        tokenizer:     Corresponding tokenizer
        train_df:      This client's local training partition (pandas DataFrame)
        val_df:        Shared validation DataFrame (IID)
        mu:            FedProx regularisation coefficient (0.0 = FedAvg)
    """

    def __init__(self, client_id, model, tokenizer, train_df, val_df, mu: float = 0.0):
        self.client_id = client_id
        self.model     = model
        self.tokenizer = tokenizer
        self.mu        = mu

        # Build PyTorch datasets from pandas DataFrames
        # Convert to HuggingFace Dataset-like dict for EmailDataset
        self.train_dataset = EmailDataset(
            {"text": train_df["text"].tolist(), "label": train_df["label"].tolist()},
            tokenizer,
        )
        self.val_dataset = EmailDataset(
            {"text": val_df["text"].tolist(), "label": val_df["label"].tolist()},
            tokenizer,
        )

        logger.info(
            f"Client {client_id} initialised | "
            f"train={len(self.train_dataset)} | val={len(self.val_dataset)} | µ={mu}"
        )

    # Flower API

    def get_parameters(self, config: dict) -> List[np.ndarray]:
        """Return current local model parameters."""
        return get_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        fit_config: dict,
    ) -> Tuple[List[np.ndarray], int, dict]:
        """
        1. Load global parameters into local model
        2. Run local training
        3. Return updated parameters + metrics
        """
        # Load global parameters
        set_parameters(self.model, parameters)

        # Allow server to override µ per round (for µ sweep experiments)
        mu = fit_config.get("mu", self.mu)

        # Local training
        _, num_examples, avg_loss = local_train(
            model         = self.model,
            train_dataset = self.train_dataset,
            num_epochs    = fit_config.get("local_epochs", config.LOCAL_EPOCHS),
            batch_size    = fit_config.get("batch_size",   config.LOCAL_BATCH),
            lr            = fit_config.get("lr",           config.LEARNING_RATE),
            mu            = mu,
        )

        logger.info(
            f"Client {self.client_id} | fit done | "
            f"examples={num_examples} | loss={avg_loss:.4f}"
        )

        return get_parameters(self.model), num_examples, {"train_loss": avg_loss}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        eval_config: dict,
    ) -> Tuple[float, int, dict]:
        """
        Load global parameters and evaluate on local validation set.
        Returns (loss, num_examples, metrics_dict) as required by Flower.
        """
        set_parameters(self.model, parameters)

        metrics = evaluate(self.model, self.val_dataset)

        logger.info(
            f"Client {self.client_id} | eval | "
            f"acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f}"
        )

        return (
            metrics["loss"],
            metrics["num_examples"],
            {
                "accuracy":  metrics["accuracy"],
                "f1":        metrics["f1"],
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
                "auc_roc":   metrics["auc_roc"],
            },
        )

# Client factory — used by the Flower simulation engine

def make_client_fn(
    partitions:  list,   # list of pd.DataFrames (one per client)
    val_df,              # shared IID val set (pd.DataFrame)
    model_key:   str,
    mu:          float = 0.0,
):
    """
    Returns a client_fn callable for fl.simulation.start_simulation().

    fl.simulation expects: client_fn(cid: str) -> fl.client.Client
    """
    # Import here to avoid circular at module load
    from models.transformer_classifier import get_model

    # Pre-load models for each client to avoid redundant downloads
    # In a large deployment this would happen on each worker; for simulation
    # we cache per-client models in a dict.
    client_models = {}

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)

        if client_id not in client_models:
            model, tokenizer = get_model(model_key)
            client_models[client_id] = (model, tokenizer)

        model, tokenizer = client_models[client_id]

        return PhishingClient(
            client_id  = client_id,
            model      = model,
            tokenizer  = tokenizer,
            train_df   = partitions[client_id],
            val_df     = val_df,
            mu         = mu,
        ).to_client()

    return client_fn