"""
aggregation/fedavg_strategy.py
-------------------------------
Custom Flower FedAvg strategy that:
  - Logs per-round aggregated metrics (accuracy, F1, AUC-ROC, etc.)
  - Stores round results in a list for CSV export
  - Implements early stopping based on validation F1 plateau
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
    EvaluateIns,
)
from flwr.server.client_proxy import ClientProxy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg with round-by-round metric logging and early stopping.

    Results are accumulated in self.round_results (list of dicts)
    and can be exported to CSV via save_results().
    """

    def __init__(
        self,
        experiment_id:  str,
        early_stop_delta:  float = config.EARLY_STOP_DELTA,
        early_stop_rounds: int   = config.EARLY_STOP_ROUNDS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experiment_id     = experiment_id
        self.early_stop_delta  = early_stop_delta
        self.early_stop_rounds = early_stop_rounds

        self.round_results: List[dict] = []
        self._best_f1        = -1.0
        self._no_improve     = 0
        self._stop_requested = False

    # Fit config broadcast

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ):
        """Broadcast fit config to all clients each round."""
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        # Inject current round so clients can log it
        updated = []
        for client, fit_ins in fit_ins_list:
            new_config = {**fit_ins.config, "round": server_round}
            updated.append((client, FitIns(fit_ins.parameters, new_config)))
        return updated

    # Aggregation

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Standard FedAvg weighted averaging."""
        if not results:
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Log per-client train losses
        losses = [r.metrics.get("train_loss", 0) for _, r in results]
        logger.info(
            f"[Round {server_round}] FedAvg fit | "
            f"clients={len(results)} | "
            f"avg_train_loss={np.mean(losses):.4f}"
        )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Weighted average of per-client evaluation metrics.
        Logs and stores results. Triggers early stop if F1 plateaus.
        """
        if not results:
            return None, {}

        # Weighted average by number of examples
        total_examples = sum(r.num_examples for _, r in results)

        def weighted_avg(metric_key):
            return sum(
                r.metrics.get(metric_key, 0) * r.num_examples
                for _, r in results
            ) / total_examples

        avg_loss      = sum(r.loss * r.num_examples for _, r in results) / total_examples
        avg_accuracy  = weighted_avg("accuracy")
        avg_f1        = weighted_avg("f1")
        avg_precision = weighted_avg("precision")
        avg_recall    = weighted_avg("recall")
        avg_auc_roc   = weighted_avg("auc_roc")

        record = {
            "experiment_id": self.experiment_id,
            "round":         server_round,
            "loss":          avg_loss,
            "accuracy":      avg_accuracy,
            "f1":            avg_f1,
            "precision":     avg_precision,
            "recall":        avg_recall,
            "auc_roc":       avg_auc_roc,
            "num_clients":   len(results),
        }
        self.round_results.append(record)

        logger.info(
            f"[Round {server_round}] FedAvg eval | "
            f"acc={avg_accuracy:.4f} | f1={avg_f1:.4f} | "
            f"auc={avg_auc_roc:.4f} | loss={avg_loss:.4f}"
        )

        # Early stopping check
        if avg_f1 > self._best_f1 + self.early_stop_delta:
            self._best_f1    = avg_f1
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self.early_stop_rounds:
                logger.info(
                    f"[Round {server_round}] Early stopping triggered — "
                    f"F1 has not improved by {self.early_stop_delta} "
                    f"for {self.early_stop_rounds} consecutive rounds."
                )
                self._stop_requested = True

        return avg_loss, {
            "accuracy":  avg_accuracy,
            "f1":        avg_f1,
            "precision": avg_precision,
            "recall":    avg_recall,
            "auc_roc":   avg_auc_roc,
        }

    def save_results(self, path: str) -> None:
        """Save round results to a CSV file."""
        import pandas as pd
        df = pd.DataFrame(self.round_results)
        df.to_csv(path, index=False)
        logger.info(f"Round results saved to {path}")
