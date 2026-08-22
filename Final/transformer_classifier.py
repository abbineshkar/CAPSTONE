"""Unified transformer utilities for federated phishing classification."""

from __future__ import annotations

import logging
import os
import random
import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible paired experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        selected = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        selected = torch.device("cpu")
        logger.info("GPU not available; using CPU.")
    return selected


DEVICE = get_device()


def get_model(model_key: str) -> tuple[nn.Module, Any]:
    """Load one configured sequence-classification model and tokenizer."""
    if model_key not in config.MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key {model_key!r}. "
            f"Valid options: {sorted(config.MODEL_CONFIGS)}"
        )

    model_config = config.MODEL_CONFIGS[model_key]
    hf_name = model_config["hf_name"]
    num_labels = model_config["num_labels"]

    logger.info("Loading model: %s", hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name,
        num_labels=num_labels,
    )
    model.to(DEVICE)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded | trainable parameters: %s", f"{parameter_count:,}")
    return model, tokenizer


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract a detached copy of the complete model state for FL transport."""
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load a complete FL state while preserving each destination tensor dtype."""
    current_state = model.state_dict()
    if len(parameters) != len(current_state):
        raise ValueError(
            f"Parameter count mismatch: received {len(parameters)}, "
            f"expected {len(current_state)}."
        )

    state_dict = OrderedDict()
    for (name, destination), array in zip(current_state.items(), parameters):
        tensor = torch.as_tensor(array, dtype=destination.dtype)
        state_dict[name] = tensor
    model.load_state_dict(state_dict, strict=True)


class EmailDataset(torch.utils.data.Dataset):
    """Tokenise an in-memory text/label mapping for PyTorch training."""

    def __init__(
        self,
        dataset: dict[str, list],
        tokenizer: Any,
        max_len: int = config.MAX_TOKEN_LEN,
    ) -> None:
        texts = dataset.get(config.TEXT_COLUMN, [])
        labels = dataset.get(config.LABEL_COLUMN, [])
        if len(texts) != len(labels):
            raise ValueError("Text and label lists must have the same length.")

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
) -> tuple[float, float]:
    """Train one epoch and return classification loss and total objective."""
    model.train()
    classification_loss_sum = 0.0
    objective_loss_sum = 0.0
    example_count = 0

    if mu > 0.0:
        if global_params is None:
            raise ValueError("FedProx requires a global-parameter snapshot.")
        global_tensors = [tensor.to(DEVICE) for tensor in global_params]
    else:
        global_tensors = None

    for batch in dataloader:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        outputs = model(**batch)
        classification_loss = outputs.loss
        objective = classification_loss

        if global_tensors is not None:
            proximal_penalty = torch.zeros((), device=DEVICE)
            for local_parameter, global_parameter in zip(
                model.parameters(), global_tensors
            ):
                proximal_penalty = proximal_penalty + torch.sum(
                    (local_parameter - global_parameter) ** 2
                )
            objective = objective + (mu / 2.0) * proximal_penalty

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.GRADIENT_CLIP_NORM
        )
        optimizer.step()
        scheduler.step()

        current_batch_size = int(batch["labels"].size(0))
        classification_loss_sum += classification_loss.item() * current_batch_size
        objective_loss_sum += objective.item() * current_batch_size
        example_count += current_batch_size

    denominator = max(example_count, 1)
    return (
        classification_loss_sum / denominator,
        objective_loss_sum / denominator,
    )


def local_train(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    num_epochs: int = config.LOCAL_EPOCHS,
    batch_size: int = config.LOCAL_BATCH,
    lr: float = config.LEARNING_RATE,
    mu: float = 0.0,
    seed: int = config.SEED,
) -> tuple[nn.Module, int, dict[str, float]]:
    """Run deterministic local training for one federated client."""
    if len(train_dataset) == 0:
        raise ValueError("Cannot train on an empty client dataset.")
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    global_params = (
        [parameter.detach().clone() for parameter in model.parameters()]
        if mu > 0.0
        else None
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=max(total_steps, 1),
    )

    classification_losses: list[float] = []
    objective_losses: list[float] = []
    for epoch in range(num_epochs):
        classification_loss, objective_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            mu=mu,
            global_params=global_params,
        )
        classification_losses.append(classification_loss)
        objective_losses.append(objective_loss)
        logger.debug(
            "Epoch %d/%d | classification_loss=%.4f | objective_loss=%.4f",
            epoch + 1,
            num_epochs,
            classification_loss,
            objective_loss,
        )

    losses = {
        "classification_loss": float(np.mean(classification_losses)),
        "objective_loss": float(np.mean(objective_losses)),
    }
    return model, len(train_dataset), losses


def evaluate(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
) -> dict[str, float | int]:
    """Evaluate a model and compute macro classification metrics."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    weighted_loss_sum = 0.0
    example_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            outputs = model(**batch)
            current_batch_size = int(batch["labels"].size(0))

            weighted_loss_sum += outputs.loss.item() * current_batch_size
            example_count += current_batch_size
            all_logits.append(outputs.logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = np.argmax(logits, axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "auc_roc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) > 1
        else 0.0,
        "loss": weighted_loss_sum / max(example_count, 1),
        "num_examples": example_count,
    }
