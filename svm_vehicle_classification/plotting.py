"""Figures for the image-feature and SVM experiment."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_examples(result):
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.5), layout="constrained")
    for row, label in enumerate((-1, 1)):
        for column, index in enumerate(np.flatnonzero(result["labels"] == label)[:4]):
            axes[row, column].imshow(result["images"][index], cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_title("Other shape" if label == -1 else "Vehicle-like shape", fontsize=10)
            axes[row, column].axis("off")
    fig.suptitle("Generated teaching images — not road photographs")
    return fig


def plot_features(result):
    index = int(np.flatnonzero(result["labels"] == 1)[0])
    image = result["images"][index]
    gy, gx = np.gradient(image)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Generated vehicle-like crop")
    axes[1].imshow(np.hypot(gx, gy), cmap="magma")
    axes[1].set_title("Gradient magnitude")
    for ax in axes[:2]:
        ax.axis("off")
    axes[2].plot(result["features"][index], lw=0.7, color="#2563eb")
    axes[2].set(title="HOG descriptor", xlabel="Feature index (1764 values)", ylabel="Normalized vote")
    return fig


def plot_confusion(result):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), layout="constrained")
    truth = result["predictions"]["actual"]
    for ax, kernel in zip(axes, ("linear", "rbf")):
        counts = confusion_matrix(truth, result["predictions"][f"SVC {kernel}"], labels=[-1, 1])
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, counts.max()))
        for (row, column), value in np.ndenumerate(counts):
            ax.text(column, row, str(value), ha="center", va="center", fontsize=14,
                    color="white" if value > counts.max() / 2 else "#1e293b")
        ax.set(xticks=[0, 1], xticklabels=["Other", "Vehicle"], yticks=[0, 1],
               yticklabels=["Other", "Vehicle"], xlabel="Predicted", ylabel="Actual", title=f"SVC {kernel}")
    return fig
