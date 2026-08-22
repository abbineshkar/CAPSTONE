"""Project-wide configuration for the 20 federated phishing experiments."""

from __future__ import annotations

import os

# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
DATA_DIR = os.path.join(BASE_DIR, "data")
NAZARIO_PATH = os.path.join(DATA_DIR, "Nazario.csv")
ENRON_PATH = os.path.join(DATA_DIR, "Enron.csv")
SPAMASSASSIN_PATH = os.path.join(DATA_DIR, "spamassassin.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_TOKEN_LEN = 128
SEED = 42
CODE_VERSION = "corrected-20exp-v1"

# Federated learning

NUM_CLIENTS = 5
NUM_ROUNDS = 20
MIN_FIT_CLIENTS = 5
MIN_EVAL_CLIENTS = 5
MIN_AVAILABLE_CLIENTS = 5
MIN_CLIENT_SAMPLES = 8
PARTITION_MAX_ATTEMPTS = 2_000

LOCAL_EPOCHS = 2
LOCAL_BATCH = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0

# Early stopping

EARLY_STOP_DELTA = 0.001
EARLY_STOP_ROUNDS = 5

# Models

MODEL_CONFIGS = {
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "num_labels": 2,
    },
    "tinybert": {
        "hf_name": "huawei-noah/TinyBERT_General_4L_312D",
        "num_labels": 2,
    },
}

# Non-IID partitioning

ALPHA_IID = 100.0
ALPHA_MODERATE = 0.5
ALPHA_SEVERE = 0.1

QUANTITY_SPLITS = [0.60, 0.20, 0.08, 0.06, 0.06]

# FedProx

FEDPROX_MU_VALUES = [0.01, 0.1, 1.0]
FEDPROX_MU_PRIMARY = 0.1

# Final 20-experiment matrix
# Tuple fields: (experiment_id, model, aggregation, condition, parameter)
# For FedAvg label-skew experiments, parameter is alpha.
# For FedProx experiments, parameter is mu; the condition selects alpha.

EXPERIMENT_MATRIX = [
    # DistilBERT + FedAvg
    ("E1", "distilbert", "fedavg", "label_skew_iid", ALPHA_IID),
    ("E2", "distilbert", "fedavg", "label_skew_moderate", ALPHA_MODERATE),
    ("E3", "distilbert", "fedavg", "label_skew_severe", ALPHA_SEVERE),
    ("E4", "distilbert", "fedavg", "feature_skew", None),

    # DistilBERT + FedProx
    ("E5", "distilbert", "fedprox", "label_skew_iid", FEDPROX_MU_PRIMARY),
    ("E6", "distilbert", "fedprox", "label_skew_moderate", FEDPROX_MU_PRIMARY),
    ("E7", "distilbert", "fedprox", "label_skew_severe", FEDPROX_MU_PRIMARY),
    ("E8", "distilbert", "fedprox", "feature_skew", FEDPROX_MU_PRIMARY),

    # TinyBERT + FedAvg
    ("E9", "tinybert", "fedavg", "label_skew_iid", ALPHA_IID),
    ("E10", "tinybert", "fedavg", "label_skew_moderate", ALPHA_MODERATE),
    ("E11", "tinybert", "fedavg", "label_skew_severe", ALPHA_SEVERE),
    ("E12", "tinybert", "fedavg", "feature_skew", None),

    # TinyBERT + FedProx
    ("E13", "tinybert", "fedprox", "label_skew_iid", FEDPROX_MU_PRIMARY),
    ("E14", "tinybert", "fedprox", "label_skew_moderate", FEDPROX_MU_PRIMARY),
    ("E15", "tinybert", "fedprox", "label_skew_severe", FEDPROX_MU_PRIMARY),
    ("E16", "tinybert", "fedprox", "feature_skew", FEDPROX_MU_PRIMARY),

    # Quantity heterogeneity
    ("E17", "distilbert", "fedavg", "quantity", None),
    ("E18", "distilbert", "fedprox", "quantity", FEDPROX_MU_PRIMARY),
    ("E19", "tinybert", "fedavg", "quantity", None),
    ("E20", "tinybert", "fedprox", "quantity", FEDPROX_MU_PRIMARY),
]

VALID_NON_IID_TYPES = {
    "label_skew_iid",
    "label_skew_moderate",
    "label_skew_severe",
    "feature_skew",
    "quantity",
}
