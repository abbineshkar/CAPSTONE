"""
aggregation/fedprox_strategy.py
--------------------------------
FedProx strategy for Flower.

FedProx (Li et al., 2020) adds a proximal regularisation term to each
client's local objective:

    F_k(w) + (µ/2) ||w - w_global||^2

The global aggregation is identical to FedAvg — the difference is purely
in the LOCAL training objective (handled in fl_client.py / local_train()).

This strategy:
  1. Broadcasts µ in the fit_config so clients know to apply the proximal term
  2. Otherwise behaves identically to LoggingFedAvg for metric logging
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitIns, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from aggregation.fedavg_strategy import LoggingFedAvg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LoggingFedProx(LoggingFedAvg):
    """
    FedProx strategy.

    Inherits all metric logging and early stopping from LoggingFedAvg.
    The key difference: µ is injected into each client's fit_config,
    which is picked up by PhishingClient.fit() and passed to local_train().

    Args:
        mu:            FedProx proximal coefficient
        experiment_id: Identifier for result CSV files
        **kwargs:      Passed through to FedAvg (fraction_fit, min_clients, etc.)
    """

    def __init__(
        self,
        mu:            float = config.FEDPROX_MU_PRIMARY,
        experiment_id: str   = "experiment",
        **kwargs,
    ):
        super().__init__(experiment_id=experiment_id, **kwargs)
        self.mu = mu
        logger.info(f"FedProx strategy initialised with µ={mu}")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Override configure_fit to inject µ into each client's config dict.
        Clients read this via fit_config["mu"] in PhishingClient.fit().
        """
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

        updated = []
        for client, fit_ins in fit_ins_list:
            new_config = {**fit_ins.config, "mu": self.mu}
            updated.append((client, FitIns(fit_ins.parameters, new_config)))

        return updated
