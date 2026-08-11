"""
config.py
---------
Central configuration for all hyperparameters, paths, and experiment settings.
FINAL VERSION — Nazario + SpamAssassin + Enron | 20 rounds | Extended Dirichlet
"""

import os

# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR  = os.path.join(BASE_DIR, "datasets")
NAZARIO_PATH = os.path.join(BASE_DIR, "data", "Nazario.csv")
ENRON_PATH   = os.path.join(BASE_DIR, "data", "Enron.csv")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset

TEXT_COLUMN  = "text"
LABEL_COLUMN = "label"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

MAX_TOKEN_LEN = 128
SEED          = 42

# Federated Learning

NUM_CLIENTS           = 5
NUM_ROUNDS            = 20  
MIN_FIT_CLIENTS       = 5
MIN_EVAL_CLIENTS      = 5
MIN_AVAILABLE_CLIENTS = 5

LOCAL_EPOCHS  = 2
LOCAL_BATCH   = 8
LEARNING_RATE = 2e-5

# Early stopping

EARLY_STOP_DELTA  = 0.001
EARLY_STOP_ROUNDS = 5

# Models

MODEL_CONFIGS = {
    "distilbert": {
        "hf_name":    "distilbert-base-uncased",
        "num_labels": 2,
    },
    "tinybert": {
        "hf_name":    "huawei-noah/TinyBERT_General_4L_312D",
        "num_labels": 2,
    },
}

# Non-IID Partitioning — Extended Dirichlet values

ALPHA_IID           = 100.0  # Near-IID baseline
ALPHA_MODERATE      = 0.5    # Moderate label skew
ALPHA_SEVERE        = 0.1    # Severe label skew
ALPHA_VERY_SEVERE   = 0.05   # Very severe label skew (new)

# Quantity heterogeneity splits

QUANTITY_SPLITS = [0.60, 0.20, 0.08, 0.06, 0.06]

# FedProx

FEDPROX_MU_VALUES  = [0.01, 0.1, 1.0]
FEDPROX_MU_PRIMARY = 0.1


# Experiment Matrix — 20 experiments 
# Original 20 + 4 new very severe skew (α=0.05)

EXPERIMENT_MATRIX = [
    # DistilBERT + FedAvg 
    ("E1",  "distilbert", "fedavg",  "label_skew_iid",        ALPHA_IID),
    ("E2",  "distilbert", "fedavg",  "label_skew_moderate",   ALPHA_MODERATE),
    ("E3",  "distilbert", "fedavg",  "label_skew_severe",     ALPHA_SEVERE),
    ("E4",  "distilbert", "fedavg",  "feature_skew",          None),

    #  DistilBERT + FedProx 
    ("E5",  "distilbert", "fedprox", "label_skew_iid",        FEDPROX_MU_PRIMARY),
    ("E6",  "distilbert", "fedprox", "label_skew_moderate",   FEDPROX_MU_PRIMARY),
    ("E7",  "distilbert", "fedprox", "label_skew_severe",     FEDPROX_MU_PRIMARY),
    ("E8",  "distilbert", "fedprox", "feature_skew",          FEDPROX_MU_PRIMARY),

    # TinyBERT + FedAvg 
    ("E9",  "tinybert",   "fedavg",  "label_skew_iid",        ALPHA_IID),
    ("E10", "tinybert",   "fedavg",  "label_skew_moderate",   ALPHA_MODERATE),
    ("E11", "tinybert",   "fedavg",  "label_skew_severe",     ALPHA_SEVERE),
    ("E12", "tinybert",   "fedavg",  "feature_skew",          None),

    # TinyBERT + FedProx 
    ("E13", "tinybert",   "fedprox", "label_skew_iid",        FEDPROX_MU_PRIMARY),
    ("E14", "tinybert",   "fedprox", "label_skew_moderate",   FEDPROX_MU_PRIMARY),
    ("E15", "tinybert",   "fedprox", "label_skew_severe",     FEDPROX_MU_PRIMARY),
    ("E16", "tinybert",   "fedprox", "feature_skew",          FEDPROX_MU_PRIMARY),

    # Quantity Heterogeneity 
    ("E17", "distilbert", "fedavg",  "quantity",              None),
    ("E18", "distilbert", "fedprox", "quantity",              FEDPROX_MU_PRIMARY),
    ("E19", "tinybert",   "fedavg",  "quantity",              None),
    ("E20", "tinybert",   "fedprox", "quantity",              FEDPROX_MU_PRIMARY),

    """# Very Severe Label Skew (α=0.05) NEW
    ("E21", "distilbert", "fedavg",  "label_skew_very_severe", ALPHA_VERY_SEVERE),
    ("E22", "distilbert", "fedprox", "label_skew_very_severe", FEDPROX_MU_PRIMARY),
    ("E23", "tinybert",   "fedavg",  "label_skew_very_severe", ALPHA_VERY_SEVERE),
    ("E24", "tinybert",   "fedprox", "label_skew_very_severe", FEDPROX_MU_PRIMARY),"""
]
