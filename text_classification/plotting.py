"""Aggregate figures only: no corpus excerpts, word clouds, or vocabulary exports."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def learning_curves(result):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, rows in result["history"].groupby("model", sort=False):
        axes[0].plot(rows.epoch, rows.train_loss, marker="o", label=name)
        axes[1].plot(rows.epoch, rows.validation_spam_f1, marker="o", label=name)
    axes[0].set(xlabel="Epoch", ylabel="Training cross-entropy", title="Recurrent model training")
    axes[1].set(xlabel="Epoch", ylabel="Spam F1", title="Validation checkpoint selection", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    return figure


def comparison(result):
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    metrics = result["metrics"].set_index("model")[["test_accuracy", "test_spam_f1"]]
    metrics.plot.barh(ax=axis)
    axis.set(xlabel="Score", ylabel="", xlim=(0, 1), title="Held-out test: accuracy and spam F1")
    axis.legend(["Accuracy", "Spam F1"], loc="upper right", bbox_to_anchor=(1, -0.18), ncol=2, fontsize=8)
    return figure


def confusion_matrices(result):
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    for axis, name in zip(axes.flat, result["metrics"].model):
        rows = result["confusion"][result["confusion"].model == name]
        matrix = rows.pivot(index="actual", columns="predicted", values="count").loc[["ham", "spam"], ["ham", "spam"]].to_numpy()
        axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set(title=name, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        for i, j in np.ndindex(2, 2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    return figure


def data_summary(result):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    metadata = result["metadata"]
    pd.DataFrame(metadata["class_counts"]).T.plot.bar(ax=axes[0], rot=0)
    axes[0].set(xlabel="Split", ylabel="Messages", title="Class imbalance after deduplication")
    histogram = metadata["train_length_histogram"]
    labels = [f"{left}–{right - 1}" for left, right in zip(histogram["bin_edges"][:-1], histogram["bin_edges"][1:])]
    axes[1].bar(labels, histogram["counts"])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(xlabel="Tokens per message", ylabel="Training messages", title="Lengths before the 60-token limit")
    return figure


def save_plots(result, output):
    for name, plot in (("learning_curves", learning_curves), ("metrics_comparison", comparison),
                       ("confusion_matrices", confusion_matrices), ("data_summary", data_summary)):
        figure = plot(result)
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)
