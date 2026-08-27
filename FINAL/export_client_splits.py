from __future__ import annotations

import os
import re
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.preprocess import load_and_merge, split_dataset
from data.partition import get_partitions


def numeric_experiment_id(exp_id: str) -> int:
    match = re.fullmatch(r"E(\d+)", str(exp_id).strip())
    return int(match.group(1)) if match else 10**9


def valid_experiment_rows() -> list[tuple[Any, ...]]:
    rows = []
    for item in config.EXPERIMENT_MATRIX:
        if not isinstance(item, (tuple, list)) or len(item) != 5:
            continue
        exp_id = str(item[0])
        match = re.fullmatch(r"E(\d+)", exp_id)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= 20:
            rows.append(tuple(item))

    rows.sort(key=lambda x: numeric_experiment_id(x[0]))

    expected = {f"E{i}" for i in range(1, 21)}
    found = {str(row[0]) for row in rows}
    missing = sorted(expected - found, key=numeric_experiment_id)
    if missing:
        raise RuntimeError(f"Missing experiments in config: {missing}")
    return rows


def condition_alpha(non_iid_type: str) -> float | None:
    if non_iid_type == "label_skew_iid":
        return float(config.ALPHA_IID)
    if non_iid_type == "label_skew_moderate":
        return float(config.ALPHA_MODERATE)
    if non_iid_type == "label_skew_severe":
        return float(config.ALPHA_SEVERE)
    return None


def experiment_mu(aggregation: str, matrix_param: Any) -> float:
    if aggregation != "fedprox":
        return 0.0
    if matrix_param is not None:
        return float(matrix_param)
    return float(getattr(config, "FEDPROX_MU_PRIMARY", 0.1))


def build_split_rows(
    experiment_id: str,
    model_key: str,
    aggregation: str,
    non_iid_type: str,
    matrix_param: Any,
    train_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    alpha = condition_alpha(non_iid_type)
    mu = experiment_mu(aggregation, matrix_param)

    partitions = get_partitions(
        train_df=train_df,
        non_iid_type=non_iid_type,
        alpha=alpha,
        num_clients=config.NUM_CLIENTS,
        seed=config.SEED,
    )

    if len(partitions) != config.NUM_CLIENTS:
        raise RuntimeError(
            f"{experiment_id}: expected {config.NUM_CLIENTS} clients, got {len(partitions)}"
        )

    label_col = getattr(config, "LABEL_COLUMN", "label")
    text_col = getattr(config, "TEXT_COLUMN", "text")
    total_training = len(train_df)

    rows = []

    for client_id, client_df in enumerate(partitions, start=1):
        total = int(len(client_df))
        positive = int((client_df[label_col] == 1).sum()) if total else 0
        negative = int((client_df[label_col] == 0).sum()) if total else 0

        if positive + negative != total:
            raise RuntimeError(
                f"{experiment_id} Client {client_id}: class counts do not sum to total"
            )

        if total:
            lengths = client_df[text_col].astype(str).str.len()
            mean_len = float(lengths.mean())
            median_len = float(lengths.median())
            min_len = int(lengths.min())
            max_len = int(lengths.max())
            positive_pct = 100 * positive / total
            negative_pct = 100 * negative / total
            training_share_pct = 100 * total / total_training
        else:
            mean_len = median_len = 0.0
            min_len = max_len = 0
            positive_pct = negative_pct = training_share_pct = 0.0

        rows.append(
            {
                "experiment_id": experiment_id,
                "model": model_key,
                "aggregation": aggregation,
                "non_iid_type": non_iid_type,
                "alpha": alpha,
                "mu": mu,
                "seed": config.SEED,
                "client_id": client_id,
                "total": total,
                "positive_phishing_or_malicious": positive,
                "negative_legitimate": negative,
                "positive_pct": round(positive_pct, 4),
                "negative_pct": round(negative_pct, 4),
                "training_share_pct": round(training_share_pct, 4),
                "mean_text_length": round(mean_len, 2),
                "median_text_length": round(median_len, 2),
                "min_text_length": min_len,
                "max_text_length": max_len,
            }
        )

    if sum(r["total"] for r in rows) != len(train_df):
        raise RuntimeError(f"{experiment_id}: client totals do not match training set")

    return rows


def call_with_optional_seed(func, *args, **kwargs):
    """Support both corrected and older function signatures."""
    try:
        return func(*args, **kwargs)
    except TypeError as exc:
        if "seed" not in str(exc):
            raise
        kwargs.pop("seed", None)
        return func(*args, **kwargs)


def main() -> None:
    out_dir = os.path.join(config.RESULTS_DIR, "client_splits")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading and preprocessing dataset...")
    full_df = call_with_optional_seed(load_and_merge, seed=config.SEED)
    train_df, val_df, test_df = call_with_optional_seed(
        split_dataset, full_df, seed=config.SEED
    )

    print(
        f"Dataset split: train={len(train_df)} | "
        f"validation={len(val_df)} | test={len(test_df)}"
    )

    all_rows = []

    for exp_id, model, aggregation, condition, param in valid_experiment_rows():
        rows = build_split_rows(
            exp_id, model, aggregation, condition, param, train_df
        )
        frame = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"{exp_id}_client_split.csv")
        frame.to_csv(path, index=False)
        all_rows.extend(rows)

        print(f"\n{exp_id} | {model} | {aggregation} | {condition}")
        print(
            frame[
                [
                    "client_id",
                    "total",
                    "positive_phishing_or_malicious",
                    "negative_legitimate",
                    "positive_pct",
                ]
            ].to_string(index=False)
        )

    master = pd.DataFrame(all_rows)
    master["_exp_num"] = master["experiment_id"].map(numeric_experiment_id)
    master = master.sort_values(["_exp_num", "client_id"]).drop(columns="_exp_num")
    master_path = os.path.join(out_dir, "all_client_splits.csv")
    master.to_csv(master_path, index=False)

    summary_rows = []
    for exp_id, group in master.groupby("experiment_id", sort=False):
        first = group.iloc[0]
        summary_rows.append(
            {
                "experiment_id": exp_id,
                "model": first["model"],
                "aggregation": first["aggregation"],
                "non_iid_type": first["non_iid_type"],
                "alpha": first["alpha"],
                "mu": first["mu"],
                "seed": first["seed"],
                "num_clients": len(group),
                "training_total": int(group["total"].sum()),
                "positive_total": int(
                    group["positive_phishing_or_malicious"].sum()
                ),
                "negative_total": int(group["negative_legitimate"].sum()),
                "smallest_client": int(group["total"].min()),
                "largest_client": int(group["total"].max()),
                "min_positive_pct": float(group["positive_pct"].min()),
                "max_positive_pct": float(group["positive_pct"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["_exp_num"] = summary["experiment_id"].map(numeric_experiment_id)
    summary = summary.sort_values("_exp_num").drop(columns="_exp_num")
    summary_path = os.path.join(out_dir, "partition_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("CLIENT SPLIT EXPORT COMPLETE")
    print(f"Per-experiment files: {out_dir}")
    print(f"Master file:          {master_path}")
    print(f"Summary file:         {summary_path}")
    print("No transformer training was run.")
    print("=" * 80)


if __name__ == "__main__":
    main()
