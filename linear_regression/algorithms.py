"""Ordinary least squares and shuffled gradient descent using NumPy only.

The objective is J(w, b) = mean((X @ w + b - y)**2) / 2.
Feature scaling is deliberately separate so it can be fitted on training data.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite 2-D array.")
    return X


def _training_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = _features(X)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
        raise ValueError("y must be a finite 1-D array with one value per row of X.")
    return X, y


@dataclass
class LinearModel:
    """Coefficients apply to the feature units used during fitting."""

    coef: np.ndarray
    intercept: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _features(X)
        if X.shape[1] != len(self.coef):
            raise ValueError("X has a different number of features from the fitted model.")
        return X @ self.coef + self.intercept


@dataclass
class GradientDescentResult:
    model: LinearModel
    # Includes the initial loss at epoch 0 when record_history=True.
    loss_history: np.ndarray
    updates_per_epoch: int


def half_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the training objective; this is half the conventional MSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Targets and predictions must be nonempty 1-D arrays of equal shape.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite.")
    return float(np.mean((y_pred - y_true) ** 2) / 2)


def least_squares(X: np.ndarray, y: np.ndarray) -> LinearModel:
    """Solve the least-squares problem with an explicit intercept column.

    NumPy's low-level least-squares solver avoids explicitly inverting X.T @ X
    and also supports rank-deficient inputs. No sklearn estimator is used here.
    """
    X, y = _training_data(X, y)
    design = np.column_stack((np.ones(len(X)), X))
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return LinearModel(coef=theta[1:].copy(), intercept=float(theta[0]))


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.01,
    epochs: int = 500,
    batch_size: int = 20,
    seed: int = 42,
    record_history: bool = True,
) -> GradientDescentResult:
    """Fit shuffled mini-batch GD; batch_size=1 gives SGD.

    One epoch visits every training sample exactly once. The final, possibly
    smaller batch is included and its gradient uses its actual size. The
    intercept is optimized as a weight on a column of ones. A local random
    generator makes repeated calls reproducible without changing global state.
    """
    X, y = _training_data(X, y)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    for name, value in [("epochs", epochs), ("batch_size", batch_size)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    design = np.column_stack((np.ones(len(X)), X))
    theta = np.zeros(design.shape[1])
    rng = np.random.default_rng(seed)
    losses = [half_mse(y, design @ theta)] if record_history else []

    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            indices = order[start : start + batch_size]
            batch = design[indices]
            error = batch @ theta - y[indices]
            gradient = batch.T @ error / len(indices)
            theta -= learning_rate * gradient
        if not np.isfinite(theta).all():
            raise FloatingPointError("Training diverged; scale features or reduce learning_rate.")
        if record_history:
            losses.append(half_mse(y, design @ theta))

    return GradientDescentResult(
        model=LinearModel(coef=theta[1:].copy(), intercept=float(theta[0])),
        loss_history=np.asarray(losses),
        updates_per_epoch=(len(X) + batch_size - 1) // batch_size,
    )
