"""Small, reusable plotting functions for the notebook and CLI."""

import matplotlib.pyplot as plt
import numpy as np

from .experiments import ExperimentResult

COLORS = {"Least squares": "#374151", "Mini-batch GD": "#2563eb", "SGD": "#d97706", "scikit-learn": "#0f766e"}


def plot_convergence(result: ExperimentResult):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    optimum = result.metrics.loc["Least squares", "train_half_mse"]
    for name in ("Mini-batch GD", "SGD"):
        values = result.histories[name].to_numpy()
        axes[0].plot(result.histories.epoch, values, label=name, color=COLORS[name])
        # Roundoff can make a numerical loss gap slightly negative.
        axes[1].plot(result.histories.epoch, np.maximum(values - optimum, 1e-12), label=name, color=COLORS[name])
    axes[0].axhline(optimum, color=COLORS["Least squares"], ls="--", label="Least-squares minimum")
    axes[0].set(yscale="log", title="Training objective", ylabel="Half MSE (log scale)")
    axes[1].set(yscale="log", title="Distance to the training optimum", ylabel="Loss gap (log scale)")
    for ax in axes:
        ax.set_xlabel("Epoch (one pass over training data)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    return fig


def plot_learning_rates(result: ExperimentResult):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name in result.learning_rates.columns[1:]:
        ax.plot(result.learning_rates.epoch, result.learning_rates[name], label=name)
    ax.axhline(result.metrics.loc["Least squares", "train_half_mse"], ls="--", color="#374151", label="Least-squares minimum")
    ax.set(xlabel="Epoch", ylabel="Training half MSE (log scale)", yscale="log", title="Learning-rate comparison on training data")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_predictions(result: ExperimentResult):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    predicted = result.models["Least squares"].predict(result.X_test)
    low = min(result.y_test.min(), predicted.min()) - 1
    high = max(result.y_test.max(), predicted.max()) + 1
    axes[0].scatter(result.y_test, predicted, color="#2563eb", alpha=0.8)
    axes[0].plot([low, high], [low, high], "--", color="#374151")
    axes[0].set(xlabel="Observed sales", ylabel="Predicted sales", title="Held-out predictions (least squares)")
    axes[1].scatter(predicted, result.y_test - predicted, color="#0f766e", alpha=0.8)
    axes[1].axhline(0, color="#374151", ls="--")
    axes[1].set(xlabel="Predicted sales", ylabel="Observed minus predicted", title="Held-out residuals")
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig


def plot_runtime(result: ExperimentResult):
    times = result.metrics.median_fit_ms.dropna()
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.barh(times.index, times, color=[COLORS[name] for name in times.index])
    ax.set(xscale="log", xlabel="Median fit time in milliseconds (log scale)", title="Fit-only runtime on this machine")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    return fig
