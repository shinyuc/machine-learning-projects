"""One experiment implementation shared by the CLI and the notebook."""

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .algorithms import LinearModel, gradient_descent, half_mse, least_squares
from .data import DEFAULT_DATA_PATH, FEATURES, TARGET, MinMaxScaler, load_advertising, split_indices


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    test_fraction: float = 0.2
    epochs: int = 500
    learning_rate: float = 0.01
    batch_size: int = 20
    timing_repeats: int = 3


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    histories: pd.DataFrame
    learning_rates: pd.DataFrame
    models: dict[str, LinearModel]
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: dict


def _fit_time_ms(fit, repeats: int) -> float:
    """Median fit time after one warmup; data preparation/plots are excluded."""
    fit()
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        fit()
        elapsed.append((perf_counter() - start) * 1000)
    return float(np.median(elapsed))


def run_experiment(
    path: str | Path = DEFAULT_DATA_PATH,
    config: ExperimentConfig = ExperimentConfig(),
) -> ExperimentResult:
    """Compare fixed optimizer settings on one 80/20 split.

    Learning-rate experiments use training loss only. The test set is reserved
    for the final fixed-setting comparison; it does not select a learning rate.
    """
    if not isinstance(config.timing_repeats, int) or config.timing_repeats < 1:
        raise ValueError("timing_repeats must be a positive integer.")
    frame = load_advertising(path)
    X, y = frame[FEATURES].to_numpy(), frame[TARGET].to_numpy()
    train, test = split_indices(len(frame), test_fraction=config.test_fraction, seed=config.seed)
    scaler = MinMaxScaler.fit(X[train])
    X_train = scaler.transform(X[train])
    y_train = y[train]

    def fit_gd(batch_size: int, record_history: bool = False):
        return gradient_descent(
            X_train, y_train, learning_rate=config.learning_rate,
            epochs=config.epochs, batch_size=batch_size, seed=config.seed,
            record_history=record_history,
        )

    def fit_sklearn():
        return LinearRegression().fit(X_train, y_train)

    mini = fit_gd(config.batch_size, True)
    sgd = fit_gd(1, True)
    reference = fit_sklearn()
    scaled_models = {
        "Least squares": least_squares(X_train, y_train),
        "Mini-batch GD": mini.model,
        "SGD": sgd.model,
        "scikit-learn": LinearModel(reference.coef_, float(reference.intercept_)),
    }
    fitters = {
        "Least squares": lambda: least_squares(X_train, y_train),
        "Mini-batch GD": lambda: fit_gd(config.batch_size),
        "SGD": lambda: fit_gd(1),
        "scikit-learn": fit_sklearn,
    }
    models = {name: scaler.to_original_scale(model) for name, model in scaled_models.items()}
    models["Mean baseline"] = LinearModel(np.zeros(len(FEATURES)), float(y_train.mean()))

    rows, coefficients = [], []
    for name, model in models.items():
        train_predictions, test_predictions = model.predict(X[train]), model.predict(X[test])
        rows.append({
            "method": name,
            "train_half_mse": half_mse(y_train, train_predictions),
            "test_rmse": float(np.sqrt(mean_squared_error(y[test], test_predictions))),
            "test_mae": float(mean_absolute_error(y[test], test_predictions)),
            "test_r2": float(r2_score(y[test], test_predictions)),
            "median_fit_ms": _fit_time_ms(fitters[name], config.timing_repeats) if name in fitters else np.nan,
        })
        coefficients.append({"method": name, "intercept": model.intercept, **dict(zip(FEATURES, model.coef))})

    histories = pd.DataFrame({
        "epoch": np.arange(config.epochs + 1),
        "Mini-batch GD": mini.loss_history,
        "SGD": sgd.loss_history,
    })
    # Fixed exploratory grid, interpreted using training loss, never test scores.
    learning_rates = pd.DataFrame({"epoch": np.arange(config.epochs + 1)})
    for rate in (0.001, 0.01, 0.1):
        result = gradient_descent(
            X_train, y_train, learning_rate=rate, epochs=config.epochs,
            batch_size=config.batch_size, seed=config.seed,
        )
        learning_rates[f"lr={rate:g}"] = result.loss_history

    metadata = {
        "dataset": "ISL Advertising", "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "config": asdict(config), "n_train": len(train), "n_test": len(test),
        "train_indices_zero_based": train.tolist(), "test_indices_zero_based": test.tolist(),
        "train_feature_minimum": scaler.minimum.tolist(), "train_feature_scale": scaler.scale.tolist(),
        "updates_per_epoch": {"Mini-batch GD": mini.updates_per_epoch, "SGD": sgd.updates_per_epoch},
        "timing": "Median fit-only milliseconds; one warmup; history disabled; excludes loading, scaling, evaluation and plotting.",
        "python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
    }
    return ExperimentResult(
        metrics=pd.DataFrame(rows).set_index("method"),
        coefficients=pd.DataFrame(coefficients).set_index("method"),
        histories=histories, learning_rates=learning_rates,
        models=models, X_test=X[test], y_test=y[test], metadata=metadata,
    )
