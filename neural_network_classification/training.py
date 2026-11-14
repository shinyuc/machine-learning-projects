"""Mini-batch training, validation-only checkpoint selection, and evaluation."""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def evaluate(model, images, labels, batch_size=256):
    """Evaluate without gradients or BatchNorm updates; return mean loss and predictions."""
    if len(images) == 0 or len(images) != len(labels):
        raise ValueError("Use a nonempty image set with one label per image.")
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    loss_sum, predicted = 0.0, []
    try:
        with torch.no_grad():
            for X, y in loader:
                logits = model(X)
                loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
                predicted.append(logits.argmax(dim=1))
    finally:
        model.train(was_training)
    predictions = torch.cat(predicted)
    return {"loss": loss_sum / len(labels),
            "accuracy": int((predictions == labels).sum()) / len(labels),
            "predictions": predictions.numpy()}


def train_model(model, train_images, train_labels, validation_images, validation_labels,
                *, epochs=20, learning_rate=0.01, batch_size=64, seed=42):
    """Train on CPU with momentum SGD and restore the lowest-validation-loss state.

    The test set is deliberately absent from this function's arguments.
    History contains full-split measurements in evaluation mode after each epoch,
    including epoch zero. BatchNorm buffers are part of the restored state.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("batch_size must be an integer of at least two.")
    if any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules()):
        if len(train_images) < 2 or len(train_images) % batch_size == 1:
            raise ValueError("BatchNorm needs at least two training rows per batch; change batch_size.")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    history, best_state, best_loss, best_epoch = [], None, float("inf"), 0

    for epoch in range(epochs + 1):
        if epoch > 0:
            model.train()
            for X, y in loader:
                optimizer.zero_grad()
                loss = loss_function(model(X), y)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite; try a smaller learning rate.")
                loss.backward()
                optimizer.step()

        training = evaluate(model, train_images, train_labels)
        validation = evaluate(model, validation_images, validation_labels)
        history.append({"epoch": epoch, "train_loss": training["loss"],
                        "train_accuracy": training["accuracy"],
                        "validation_loss": validation["loss"],
                        "validation_accuracy": validation["accuracy"]})
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "history": history, "best_epoch": best_epoch}
