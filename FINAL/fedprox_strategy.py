from __future__ import annotations

import logging
import os
import sys

from flwr.common import FitIns, Parameters

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from aggregation.fedavg_strategy import LoggingFedAvg

logger = logging.getLogger(__name__)


class LoggingFedProx(LoggingFedAvg):
    """Broadcast the configured FedProx coefficient to every client."""

    def __init__(
        self,
        mu: float = config.FEDPROX_MU_PRIMARY,
        experiment_id: str = "experiment",
        **kwargs,
    ) -> None:
        if mu < 0.0:
            raise ValueError("FedProx mu must be non-negative.")
        super().__init__(experiment_id=experiment_id, **kwargs)
        self.mu = float(mu)
        logger.info("FedProx strategy initialised with mu=%s", self.mu)

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
            new_config = {**fit_ins.config, "mu": self.mu}
            updated.append((client, FitIns(fit_ins.parameters, new_config)))
        return updated
