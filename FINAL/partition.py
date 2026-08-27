from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def _validate_input(train_df: pd.DataFrame, num_clients: int) -> None:
    required = {config.TEXT_COLUMN, config.LABEL_COLUMN}
    missing = required.difference(train_df.columns)
    if missing:
        raise ValueError(f"Training DataFrame is missing columns: {sorted(missing)}")
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if len(train_df) < num_clients:
        raise ValueError(
            f"Cannot partition {len(train_df)} rows across {num_clients} clients."
        )


def _log_partition_stats(partitions: Sequence[pd.DataFrame], label: str = "") -> None:
    logger.info("Partition stats [%s]:", label)
    for i, frame in enumerate(partitions):
        n = len(frame)
        n_phish = int(frame[config.LABEL_COLUMN].sum()) if n else 0
        n_legit = n - n_phish
        if n:
            logger.info(
                "  Client %d: total=%d | phishing=%d (%.1f%%) | legit=%d (%.1f%%)",
                i + 1,
                n,
                n_phish,
                100.0 * n_phish / n,
                n_legit,
                100.0 * n_legit / n,
            )
        else:
            logger.info("  Client %d: total=0 | EMPTY", i + 1)


def _largest_remainder_counts(total: int, fractions: Sequence[float]) -> np.ndarray:
    """Allocate an integer total according to fractions without losing rows."""
    raw = np.asarray(fractions, dtype=float) * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def partition_label_skew(
    train_df: pd.DataFrame,
    num_clients: int = config.NUM_CLIENTS,
    alpha: float = config.ALPHA_MODERATE,
    seed: int = config.SEED,
    min_client_samples: int = config.MIN_CLIENT_SAMPLES,
    max_attempts: int = config.PARTITION_MAX_ATTEMPTS,
) -> list[pd.DataFrame]:
    """Create label-distribution skew using a Dirichlet allocation.

    The allocation is resampled until every client has at least
    ``min_client_samples`` total rows. Clients may still contain only one class,
    which is intentional under severe label skew.
    """
    _validate_input(train_df, num_clients)
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")
    if min_client_samples < 1:
        raise ValueError("min_client_samples must be at least 1")
    if len(train_df) < num_clients * min_client_samples:
        raise ValueError(
            "Dataset is too small for the requested minimum client size: "
            f"{len(train_df)} < {num_clients * min_client_samples}."
        )

    classes = sorted(train_df[config.LABEL_COLUMN].unique().tolist())
    rng = np.random.default_rng(seed)

    for attempt in range(1, max_attempts + 1):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]

        for cls in classes:
            cls_indices = train_df.index[
                train_df[config.LABEL_COLUMN] == cls
            ].to_numpy(copy=True)
            rng.shuffle(cls_indices)

            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
            counts = rng.multinomial(len(cls_indices), proportions)
            offsets = np.concatenate(([0], np.cumsum(counts)))

            for client_id in range(num_clients):
                start, end = offsets[client_id], offsets[client_id + 1]
                client_indices[client_id].extend(cls_indices[start:end].tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) < min_client_samples:
            continue

        partitions = []
        for client_id, indices in enumerate(client_indices):
            client_df = train_df.loc[indices].copy()
            client_df = client_df.sample(
                frac=1.0,
                random_state=seed + client_id,
            ).reset_index(drop=True)
            partitions.append(client_df)

        _log_partition_stats(
            partitions,
            label=f"label_skew alpha={alpha} attempt={attempt}",
        )
        return partitions

    raise RuntimeError(
        "Could not generate a Dirichlet partition satisfying the minimum "
        f"client size after {max_attempts} attempts. Increase the dataset size, "
        "reduce MIN_CLIENT_SAMPLES, or increase alpha."
    )


def partition_feature_skew(
    train_df: pd.DataFrame,
    num_clients: int = config.NUM_CLIENTS,
    seed: int = config.SEED,
) -> list[pd.DataFrame]:
    """Partition phishing emails into text-length bands across clients."""
    _validate_input(train_df, num_clients)

    phishing_df = train_df[train_df[config.LABEL_COLUMN] == 1].copy()
    legit_df = train_df[train_df[config.LABEL_COLUMN] == 0].copy()

    if len(phishing_df) < num_clients or len(legit_df) < num_clients:
        raise ValueError("Feature skew requires at least one row per class per client.")

    phishing_df["_text_len"] = phishing_df[config.TEXT_COLUMN].str.len()
    phishing_df = (
        phishing_df.sort_values("_text_len")
        .drop(columns="_text_len")
        .reset_index(drop=True)
    )
    legit_df = legit_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    phish_boundaries = np.linspace(0, len(phishing_df), num_clients + 1, dtype=int)
    legit_boundaries = np.linspace(0, len(legit_df), num_clients + 1, dtype=int)
    phish_splits = [
        phishing_df.iloc[phish_boundaries[i]:phish_boundaries[i + 1]].copy()
        for i in range(num_clients)
    ]
    legit_splits = [
        legit_df.iloc[legit_boundaries[i]:legit_boundaries[i + 1]].copy()
        for i in range(num_clients)
    ]

    partitions = []
    for client_id in range(num_clients):
        client_df = pd.concat(
            [phish_splits[client_id], legit_splits[client_id]],
            ignore_index=True,
        )
        client_df = client_df.sample(
            frac=1.0,
            random_state=seed + client_id,
        ).reset_index(drop=True)
        partitions.append(client_df)

    _log_partition_stats(partitions, label="feature_skew")
    return partitions


def partition_quantity(
    train_df: pd.DataFrame,
    splits: Sequence[float] = config.QUANTITY_SPLITS,
    seed: int = config.SEED,
) -> list[pd.DataFrame]:
    """Create quantity heterogeneity while preserving class proportions."""
    num_clients = len(splits)
    _validate_input(train_df, num_clients)

    fractions = np.asarray(splits, dtype=float)
    if np.any(fractions <= 0):
        raise ValueError("All quantity split fractions must be greater than zero.")
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"Quantity split fractions must sum to 1.0, got {fractions.sum()}.")

    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in sorted(train_df[config.LABEL_COLUMN].unique().tolist()):
        cls_indices = train_df.index[
            train_df[config.LABEL_COLUMN] == cls
        ].to_numpy(copy=True)
        rng.shuffle(cls_indices)
        counts = _largest_remainder_counts(len(cls_indices), fractions)
        offsets = np.concatenate(([0], np.cumsum(counts)))

        for client_id in range(num_clients):
            start, end = offsets[client_id], offsets[client_id + 1]
            client_indices[client_id].extend(cls_indices[start:end].tolist())

    partitions = []
    for client_id, indices in enumerate(client_indices):
        client_df = train_df.loc[indices].copy()
        client_df = client_df.sample(
            frac=1.0,
            random_state=seed + client_id,
        ).reset_index(drop=True)
        partitions.append(client_df)

    if sum(map(len, partitions)) != len(train_df):
        raise RuntimeError("Quantity partitioning lost or duplicated rows.")

    _log_partition_stats(partitions, label="quantity_heterogeneity")
    return partitions


def get_partitions(
    train_df: pd.DataFrame,
    non_iid_type: str,
    alpha: float | None = None,
    num_clients: int = config.NUM_CLIENTS,
    seed: int = config.SEED,
) -> list[pd.DataFrame]:
    """Route to one of the five conditions used in E1-E20."""
    if non_iid_type == "label_skew_iid":
        selected_alpha = config.ALPHA_IID if alpha is None else alpha
        return partition_label_skew(
            train_df,
            num_clients=num_clients,
            alpha=selected_alpha,
            seed=seed,
        )
    if non_iid_type == "label_skew_moderate":
        selected_alpha = config.ALPHA_MODERATE if alpha is None else alpha
        return partition_label_skew(
            train_df,
            num_clients=num_clients,
            alpha=selected_alpha,
            seed=seed,
        )
    if non_iid_type == "label_skew_severe":
        selected_alpha = config.ALPHA_SEVERE if alpha is None else alpha
        return partition_label_skew(
            train_df,
            num_clients=num_clients,
            alpha=selected_alpha,
            seed=seed,
        )
    if non_iid_type == "feature_skew":
        return partition_feature_skew(train_df, num_clients=num_clients, seed=seed)
    if non_iid_type == "quantity":
        if num_clients != len(config.QUANTITY_SPLITS):
            raise ValueError(
                "Quantity heterogeneity is defined for exactly "
                f"{len(config.QUANTITY_SPLITS)} clients."
            )
        return partition_quantity(train_df, splits=config.QUANTITY_SPLITS, seed=seed)

    raise ValueError(
        f"Unknown non_iid_type: {non_iid_type!r}. "
        f"Choose from: {', '.join(sorted(config.VALID_NON_IID_TYPES))}."
    )
