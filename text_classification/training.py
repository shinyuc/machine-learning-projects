"""Mini-batch Adam training; select epochs by validation spam F1 only."""

from copy import deepcopy
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def classification_metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "spam_precision": float(precision),
            "spam_recall": float(recall), "spam_f1": float(f1)}


def make_loader(data, indices, batch_size=16, shuffle=False, seed=42):
    indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    dataset = TensorDataset(data["tokens"][indices], data["lengths"][indices],
                            torch.as_tensor(data["labels"])[indices])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model, data, indices, batch_size=128):
    model.eval()
    predictions, actual, loss_sum = [], [], 0.0
    for tokens, lengths, labels in make_loader(data, indices, batch_size):
        logits = model(tokens, lengths)
        loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
        predictions.extend(logits.argmax(dim=1).tolist())
        actual.extend(labels.tolist())
    return {"loss": loss_sum / len(actual), **classification_metrics(actual, predictions),
            "confusion": confusion_matrix(actual, predictions, labels=[0, 1])}


def train_model(model, data, learning_rate, epochs=5, batch_size=16, seed=42):
    if epochs < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError("Epochs, learning rate and batch size must be positive.")
    loader = make_loader(data, data["train"], batch_size, shuffle=True, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_f1, best_epoch, best_state = -1.0, 0, None
    history = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for tokens, lengths, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(tokens, lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; no results were exported.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        validation = evaluate(model, data, data["validation"])
        history.append({"epoch": epoch, "train_loss": loss_sum / len(data["train"]),
                        **{f"validation_{key}": value for key, value in validation.items() if key != "confusion"}})
        if validation["spam_f1"] > best_f1:
            best_f1, best_epoch, best_state = validation["spam_f1"], epoch, deepcopy(model.state_dict())
        print(f"  epoch {epoch}/{epochs}: loss={history[-1]['train_loss']:.4f}, "
              f"validation spam F1={validation['spam_f1']:.4f}", flush=True)
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "training_seconds": perf_counter() - started}
