"""
models/transformer_classifier.py
---------------------------------
Wraps DistilBERT and TinyBERT from HuggingFace into a unified interface
for federated training.

Provides:
  - get_model()       → returns (model, tokenizer) for a given model key
  - get_parameters()  → extract numpy weights for FL communication
  - set_parameters()  → load numpy weights back into model
  - train_one_epoch() → single local training epoch
  - evaluate()        → full evaluation returning dict of metrics
"""

import os
import sys
import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Device

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available — using CPU")
    return device


DEVICE = get_device()

# Model + Tokenizer factory

def get_model(model_key: str):
    """
    Load a pre-trained transformer model and its tokenizer.

    Args:
        model_key: "distilbert" or "tinybert"

    Returns:
        (model, tokenizer)
    """
    if model_key not in config.MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Valid options: {list(config.MODEL_CONFIGS.keys())}"
        )

    cfg = config.MODEL_CONFIGS[model_key]
    hf_name    = cfg["hf_name"]
    num_labels = cfg["num_labels"]

    logger.info(f"Loading model: {hf_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=num_labels
    )
    model.to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded | trainable params: {param_count:,}")
    return model, tokenizer

# FL parameter serialisation

def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of numpy arrays (for Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load aggregated numpy parameters back into a model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)

# HuggingFace Dataset → PyTorch Dataset helper

class EmailDataset(torch.utils.data.Dataset):
    """
    Wraps a tokenized HuggingFace Dataset for use with PyTorch DataLoader.
    """
    def __init__(self, hf_dataset, tokenizer, max_len: int = config.MAX_TOKEN_LEN):
        self.encodings = tokenizer(
            hf_dataset["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(hf_dataset["label"], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# Training

def train_one_epoch(
    model:       nn.Module,
    dataloader:  DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    mu:          float = 0.0,
    global_params: Optional[list[torch.Tensor]] = None,
) -> float:
    """
    Train the model for one epoch.

    If mu > 0 and global_params is provided, FedProx proximal term is added:
        loss += (mu / 2) * ||w - w_global||^2

    Args:
        model:         Local model
        dataloader:    Training DataLoader
        optimizer:     AdamW optimiser
        scheduler:     Learning rate scheduler
        mu:            FedProx regularisation coefficient (0 = FedAvg)
        global_params: Snapshot of global model parameters (before local training)

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    # Snapshot global params onto device once
    if mu > 0 and global_params is not None:
        global_tensors = [p.to(DEVICE) for p in global_params]
    else:
        global_tensors = None

    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss

        # FedProx proximal regularisation
        if mu > 0 and global_tensors is not None:
            prox_loss = 0.0
            for local_p, global_p in zip(model.parameters(), global_tensors):
                prox_loss += torch.norm(local_p - global_p) ** 2
            loss = loss + (mu / 2.0) * prox_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def local_train(
    model:         nn.Module,
    train_dataset: torch.utils.data.Dataset,
    num_epochs:    int   = config.LOCAL_EPOCHS,
    batch_size:    int   = config.LOCAL_BATCH,
    lr:            float = config.LEARNING_RATE,
    mu:            float = 0.0,
) -> tuple[nn.Module, int, float]:
    """
    Full local training loop for a federated client.

    Captures a snapshot of the global model params before training
    (required for FedProx proximal term).

    Returns:
        (trained_model, num_examples, avg_loss)
    """
    # Snapshot global parameters for FedProx
    global_params = [p.detach().clone() for p in model.parameters()]

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # Keep 0 for Flower simulation compatibility
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler,
            mu=mu, global_params=global_params,
        )
        total_loss += epoch_loss
        logger.debug(f"  Epoch {epoch+1}/{num_epochs} — loss: {epoch_loss:.4f}")

    return model, len(train_dataset), total_loss / num_epochs

# Evaluation

def evaluate(
    model:    nn.Module,
    dataset:  torch.utils.data.Dataset,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate model on a dataset and return a dict of metrics:
      accuracy, f1, precision, recall, auc_roc, loss

    Returns:
        dict with float values and num_examples (int)
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)

            total_loss += outputs.loss.item()
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds  = np.argmax(logits, axis=1)
    probs  = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="macro", zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "loss":      total_loss / max(len(dataloader), 1),
        "num_examples": len(labels),
    }
    return metrics
