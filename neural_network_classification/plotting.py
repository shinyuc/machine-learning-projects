"""Plots of aggregate measurements; no source digit images are exported."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_learning_curves(result):
    groups = list(result["history"].groupby("model", sort=False))
    fig, axes = plt.subplots(2, len(groups), figsize=(15, 6), sharex=True, layout="constrained")
    for column, (name, group) in enumerate(groups):
        axes[0, column].set_title(name.replace(" (lr=0.1)", "\nlr=0.1"), fontsize=10)
        for split, color in [("train", "#2563eb"), ("validation", "#d97706")]:
            axes[0, column].plot(group.epoch, group[f"{split}_loss"], label=split, color=color)
            axes[1, column].plot(group.epoch, group[f"{split}_accuracy"], label=split, color=color)
        best = result["metrics"].set_index("model").loc[name, "best_epoch"]
        for ax in axes[:, column]:
            ax.axvline(best, ls=":", color="#64748b", lw=1, label="Selected epoch")
            ax.grid(alpha=0.2)
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].set(ylim=(0, 1.03), xlabel="Epoch")
    axes[0, 0].set_ylabel("Cross-entropy")
    axes[1, 0].set_ylabel("Accuracy")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training and validation curves — no test-set monitoring")
    return fig


def plot_accuracy(result):
    table = result["metrics"]
    positions = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for offset, split, color in [(-0.18, "validation", "#64748b"), (0.18, "test", "#2563eb")]:
        bars = ax.bar(positions + offset, table[f"{split}_accuracy"], width=0.35, label=split, color=color)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in table[f"{split}_accuracy"]], fontsize=8, padding=3)
    ax.axhline(result["metadata"]["majority_baseline"]["test_accuracy"], ls="--", color="#b45309",
               label="Training-majority test baseline")
    ax.set(xticks=positions, xticklabels=[name.replace(" + BN", "\n+ BN").replace(" (lr=0.1)", "\n(lr=0.1)") for name in table.model],
           ylim=(0, 1.23), ylabel="Accuracy", title="Fixed configurations at their validation-selected checkpoints")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    return fig


def plot_confusion(result):
    name = result["metadata"]["selected_model"]
    counts = confusion_matrix(result["predictions"]["actual"], result["predictions"][name], labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(6.5, 5.8), layout="constrained")
    ax.imshow(counts, cmap="Blues", vmin=0)
    for (row, column), count in np.ndenumerate(counts):
        ax.text(column, row, str(count), ha="center", va="center", fontsize=9,
                color="white" if count > counts.max() / 2 else "#1e293b")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="Actual digit",
           title=f"Test predictions: {name}\nModel chosen by validation loss")
    return fig


def plot_class_counts(result):
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    labels = result["data"]["labels"].numpy()
    for offset, split in zip([-0.25, 0, 0.25], ["train", "validation", "test"]):
        counts = np.bincount(labels[result["data"][split]], minlength=10)
        ax.bar(np.arange(10) + offset, counts, width=0.25, label=split)
    ax.set(xticks=range(10), ylim=(0, 140), xlabel="Digit", ylabel="Images", title="Stratified 60/20/20 split")
    ax.legend(loc="upper center", ncol=3)
    return fig
