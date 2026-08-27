from __future__ import annotations

import copy
import logging
import os
import sys
from typing import Optional

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with round metrics and reproducible best-checkpoint selection."""

    def __init__(
        self,
        experiment_id: str,
        early_stop_delta: float = config.EARLY_STOP_DELTA,
        early_stop_rounds: int = config.EARLY_STOP_ROUNDS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.experiment_id = experiment_id
        self.early_stop_delta = early_stop_delta
        self.early_stop_rounds = early_stop_rounds

        self.round_results: list[dict] = []
        self.latest_parameters: Parameters | None = None
        self.best_parameters: Parameters | None = None
        self.best_round = 0
        self.best_f1 = float("-inf")
        self.no_improve = 0
        self.plateau_detected_round: int | None = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ):
        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        updated = []
        for client, fit_ins in fit_instructions:
            new_config = {
                **fit_ins.config,
                "round": server_round,
                "local_epochs": config.LOCAL_EPOCHS,
                "batch_size": config.LOCAL_BATCH,
                "lr": config.LEARNING_RATE,
            }
            updated.append((client, FitIns(fit_ins.parameters, new_config)))
        return updated

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.latest_parameters = copy.deepcopy(aggregated_parameters)

        classification_losses = [
            float(result.metrics.get("classification_loss", 0.0))
            for _, result in results
        ]
        objective_losses = [
            float(result.metrics.get("objective_loss", 0.0))
            for _, result in results
        ]
        logger.info(
            "[Round %d] fit | clients=%d | classification_loss=%.4f "
            "| objective_loss=%.4f",
            server_round,
            len(results),
            float(np.mean(classification_losses)),
            float(np.mean(objective_losses)),
        )
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(result.num_examples for _, result in results)
        if total_examples <= 0:
            raise ValueError("Evaluation returned zero examples.")

        def weighted_average(metric_key: str) -> float:
            return sum(
                float(result.metrics.get(metric_key, 0.0)) * result.num_examples
                for _, result in results
            ) / total_examples

        average_loss = sum(
            float(result.loss) * result.num_examples for _, result in results
        ) / total_examples
        metrics = {
            "accuracy": weighted_average("accuracy"),
            "f1": weighted_average("f1"),
            "precision": weighted_average("precision"),
            "recall": weighted_average("recall"),
            "auc_roc": weighted_average("auc_roc"),
        }

        self.round_results.append(
            {
                "experiment_id": self.experiment_id,
                "round": server_round,
                "loss": average_loss,
                **metrics,
                "num_clients": len(results),
            }
        )

        logger.info(
            "[Round %d] validation | accuracy=%.4f | macro_f1=%.4f "
            "| auc=%.4f | loss=%.4f",
            server_round,
            metrics["accuracy"],
            metrics["f1"],
            metrics["auc_roc"],
            average_loss,
        )

        if metrics["f1"] > self.best_f1 + self.early_stop_delta:
            self.best_f1 = metrics["f1"]
            self.best_round = server_round
            self.no_improve = 0
            if self.latest_parameters is not None:
                self.best_parameters = copy.deepcopy(self.latest_parameters)
        else:
            self.no_improve += 1
            if (
                self.no_improve >= self.early_stop_rounds
                and self.plateau_detected_round is None
            ):
                self.plateau_detected_round = server_round
                logger.info(
                    "[Round %d] validation plateau detected after %d "
                    "non-improving rounds.",
                    server_round,
                    self.early_stop_rounds,
                )

        return average_loss, metrics

    def parameters_for_testing(self) -> Parameters:
        """Return the best validation checkpoint, falling back to the latest."""
        selected = (
            self.best_parameters
            if self.best_parameters is not None
            else self.latest_parameters
        )
        if selected is None:
            raise RuntimeError("No aggregated parameters are available for testing.")
        return selected

    def save_results(self, path: str) -> None:
        import pandas as pd

        pd.DataFrame(self.round_results).to_csv(path, index=False)
        logger.info("Round results saved to %s", path)
