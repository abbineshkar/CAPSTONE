"""Dataset loading, cleaning, deduplication, balancing, and splitting.

Positive class (label=1): Nazario corpora plus SpamAssassin spam.
Negative class (label=0): Enron plus SpamAssassin ham.

This preserves the original experimental class definition. Exact duplicates are
removed after cleaning, and the classes are balanced only after that filtering.
"""

from __future__ import annotations

import logging
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

_HEADER_PATTERN = re.compile(
    r"^(From|To|Cc|Bcc|Subject|Date|Message-ID|MIME|Content|"
    r"Received|Return-Path|X-[\w-]+):.*$",
    flags=re.MULTILINE | re.IGNORECASE,
)


def clean_text(text: object) -> str:
    """Normalise one email body while retaining basic punctuation."""
    if not isinstance(text, str):
        return ""
    text = _HEADER_PATTERN.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_required_csv(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} dataset not found: {path}")
    return pd.read_csv(path)


def load_and_merge(seed: int = config.SEED) -> pd.DataFrame:
    """Load all sources and return a cleaned, deduplicated, balanced dataset."""
    naz1 = _read_required_csv(config.NAZARIO_PATH, "Nazario")
    naz2_path = os.path.join(config.DATA_DIR, "Nazario_5.csv")
    naz2 = _read_required_csv(naz2_path, "Nazario_5")

    if "body" not in naz1.columns or "body" not in naz2.columns:
        raise ValueError("Nazario CSV files must contain a 'body' column.")

    naz1 = naz1[["body"]].rename(columns={"body": config.TEXT_COLUMN})
    naz2 = naz2[["body"]].rename(columns={"body": config.TEXT_COLUMN})

    if os.path.exists(config.SPAMASSASSIN_PATH):
        spamassassin = pd.read_csv(config.SPAMASSASSIN_PATH)
        required = {config.TEXT_COLUMN, config.LABEL_COLUMN}
        missing = required.difference(spamassassin.columns)
        if missing:
            raise ValueError(
                "spamassassin.csv is missing columns: " f"{sorted(missing)}"
            )
        spam_only = spamassassin.loc[
            spamassassin[config.LABEL_COLUMN] == 1,
            [config.TEXT_COLUMN],
        ].copy()
        ham_only = spamassassin.loc[
            spamassassin[config.LABEL_COLUMN] == 0,
            [config.TEXT_COLUMN],
        ].copy()
        logger.info(
            "SpamAssassin spam=%d | ham=%d", len(spam_only), len(ham_only)
        )
    else:
        spam_only = pd.DataFrame(columns=[config.TEXT_COLUMN])
        ham_only = pd.DataFrame(columns=[config.TEXT_COLUMN])
        logger.warning(
            "spamassassin.csv not found; continuing with Nazario and Enron only."
        )

    phishing = pd.concat([naz1, naz2, spam_only], ignore_index=True)
    phishing[config.LABEL_COLUMN] = 1

    enron = _read_required_csv(config.ENRON_PATH, "Enron")
    if "message" not in enron.columns:
        raise ValueError("Enron.csv must contain a 'message' column.")
    enron = enron[["message"]].rename(columns={"message": config.TEXT_COLUMN})

    legitimate = pd.concat([enron, ham_only], ignore_index=True)
    legitimate[config.LABEL_COLUMN] = 0

    logger.info(
        "Raw rows: positive=%d | negative=%d", len(phishing), len(legitimate)
    )

    combined = pd.concat([phishing, legitimate], ignore_index=True)
    combined[config.TEXT_COLUMN] = combined[config.TEXT_COLUMN].apply(clean_text)
    combined = combined.loc[combined[config.TEXT_COLUMN].str.len() > 20].copy()

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=[config.TEXT_COLUMN]).reset_index(drop=True)
    logger.info("Removed %d exact duplicate cleaned texts.", before_dedup - len(combined))

    phishing_clean = combined.loc[combined[config.LABEL_COLUMN] == 1]
    legitimate_clean = combined.loc[combined[config.LABEL_COLUMN] == 0]
    n_per_class = min(len(phishing_clean), len(legitimate_clean))
    if n_per_class == 0:
        raise ValueError("At least one cleaned sample is required in each class.")

    phishing_clean = phishing_clean.sample(n=n_per_class, random_state=seed)
    legitimate_clean = legitimate_clean.sample(n=n_per_class, random_state=seed)

    dataset = pd.concat(
        [phishing_clean, legitimate_clean], ignore_index=True
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Final dataset: rows=%d | positive=%d | negative=%d",
        len(dataset),
        int(dataset[config.LABEL_COLUMN].sum()),
        int((dataset[config.LABEL_COLUMN] == 0).sum()),
    )
    return dataset


def split_dataset(
    df: pd.DataFrame,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a deterministic stratified 70/15/15 train/validation/test split."""
    required = {config.TEXT_COLUMN, config.LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    test_plus_val = config.VAL_RATIO + config.TEST_RATIO
    if not 0.0 < test_plus_val < 1.0:
        raise ValueError("VAL_RATIO + TEST_RATIO must be between 0 and 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=test_plus_val,
        stratify=df[config.LABEL_COLUMN],
        random_state=seed,
    )
    relative_test_size = config.TEST_RATIO / test_plus_val
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[config.LABEL_COLUMN],
        random_state=seed,
    )

    logger.info(
        "Split: train=%d | validation=%d | test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_vocabulary_overlap(client_texts: list[list[str]]) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity between client vocabularies."""

    def vocabulary(texts: list[str]) -> set[str]:
        words: set[str] = set()
        for text in texts:
            words.update(text.lower().split())
        return words

    vocabularies = [vocabulary(texts) for texts in client_texts]
    n_clients = len(vocabularies)
    matrix = [[0.0] * n_clients for _ in range(n_clients)]

    for i in range(n_clients):
        for j in range(n_clients):
            intersection = len(vocabularies[i] & vocabularies[j])
            union = len(vocabularies[i] | vocabularies[j])
            matrix[i][j] = intersection / union if union else 0.0

    labels = [f"Client {i + 1}" for i in range(n_clients)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


if __name__ == "__main__":
    frame = load_and_merge()
    train, validation, test = split_dataset(frame)
    print("Dataset ready")
    print(f"Train: {len(train)} | Validation: {len(validation)} | Test: {len(test)}")