# Federated Phishing Email Detection — Full Pipeline
---

## Project Structure

```
federated_phishing/
├── README.md
├── requirements.txt
├── config.py                    # Central config for all hyperparameters
├── data/
│   ├── preprocess.py            # Load, clean, merge Nazario + Enron
│   └── partition.py             # Dirichlet Non-IID partitioning (label, feature, quantity)
├── models/
│   └── transformer_classifier.py  # DistilBERT / TinyBERT classifier wrappers
├── clients/
│   └── fl_client.py             # Flower client (local train + eval)
├── aggregation/
│   ├── fedavg_strategy.py       # FedAvg strategy
│   └── fedprox_strategy.py      # FedProx strategy with proximal term
├── experiments/
│   └── run_experiment.py        # Single experiment runner (E1–E16)
├── run_all_experiments.py        # Runs full 2×2×3 matrix and saves results
├── results/                     # CSVs, per-round logs saved here
└── notebooks/
    └── analysis.ipynb           # Full analysis + plots notebook
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your datasets
Put your CSV files in a `datasets/` folder at the project root:
```
datasets/
├── nazario.csv      # columns: text, label=1
└── enron.csv        # columns: text, label=0
```
> Nazario from Zenodo DOI: 10.5281/zenodo.8339691  
> Enron: any cleaned version with a `text` column

### 3. Run all 16 experiments
```bash
python run_all_experiments.py
```
Results are saved to `results/`.

### 4. Run a single experiment (E.g., E1: DistilBERT + FedAvg + IID baseline)
```bash
python experiments/run_experiment.py \
  --model distilbert \
  --aggregation fedavg \
  --non_iid_type label_skew \
  --alpha 100 \
  --experiment_id E1
```

### 5. Open the analysis notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Experiment Matrix (16 runs)

| ID  | Model      | Aggregation | Non-IID Condition        | α / µ  |
|-----|------------|-------------|--------------------------|--------|
| E1  | DistilBERT | FedAvg      | IID Baseline             | α=100  |
| E2  | DistilBERT | FedAvg      | Label Skew (moderate)    | α=0.5  |
| E3  | DistilBERT | FedAvg      | Label Skew (severe)      | α=0.1  |
| E4  | DistilBERT | FedAvg      | Feature Skew             | —      |
| E5  | DistilBERT | FedProx     | IID Baseline             | µ=0.1  |
| E6  | DistilBERT | FedProx     | Label Skew (moderate)    | µ=0.1  |
| E7  | DistilBERT | FedProx     | Label Skew (severe)      | µ=0.1  |
| E8  | DistilBERT | FedProx     | Feature Skew             | µ=0.1  |
| E9  | TinyBERT   | FedAvg      | IID Baseline             | α=100  |
| E10 | TinyBERT   | FedAvg      | Label Skew (moderate)    | α=0.5  |
| E11 | TinyBERT   | FedAvg      | Label Skew (severe)      | α=0.1  |
| E12 | TinyBERT   | FedAvg      | Feature Skew             | —      |
| E13 | TinyBERT   | FedProx     | IID Baseline             | µ=0.1  |
| E14 | TinyBERT   | FedProx     | Label Skew (moderate)    | µ=0.1  |
| E15 | TinyBERT   | FedProx     | Label Skew (severe)      | µ=0.1  |
| E16 | TinyBERT   | FedProx     | Feature Skew             | µ=0.1  |

> Quantity heterogeneity experiments use 60/20/8/6/6% power-law splits (appended after E16).

---

## Metrics Recorded Per Round
- Accuracy, F1-score (macro), AUC-ROC, Precision, Recall
- Convergence round (first round where val-F1 plateaus within 0.001)
