"""Two figures shared by the command-line experiment and notebook."""

import matplotlib.pyplot as plt
import numpy as np


def plot_tree(tree):
    """Place leaves left to right and each parent above its children."""
    positions = {}
    edges = []
    leaf_count = 0

    def place(node, depth=0):
        nonlocal leaf_count
        child_x = []
        for value, child in node["children"].items():
            child_x.append(place(child, depth + 1))
            edges.append((node, child, value))
        if child_x:
            x = sum(child_x) / len(child_x)
        else:
            x = leaf_count
            leaf_count += 1
        positions[id(node)] = (x, -depth, node)
        return x

    place(tree)
    depth = max(-y for _, y, _ in positions.values())
    fig, ax = plt.subplots(figsize=(max(7, leaf_count * 2.2), max(3, (depth + 1) * 1.6)))
    for parent, child, value in edges:
        x1, y1, _ = positions[id(parent)]
        x2, y2, _ = positions[id(child)]
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(value),
                ha="center", va="center", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    for x, y, node in positions.values():
        leaf = node["feature"] is None
        label = f"predict: {node['prediction']}" if leaf else f"split: {node['feature']}"
        color = "#dcfce7" if leaf else "#dbeafe"
        ax.text(x, y, f"{label}\nn = {node['samples']}", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "#64748b"},
                zorder=3)
    ax.set(xlim=(-0.7, leaf_count - 0.3), ylim=(-depth - 0.5, 0.5), title="ID3 learned from training data")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(predictions):
    """Compare held-out labels and the full ID3 tree's predictions."""
    labels = ["indoor", "walk"]
    actual = predictions["actual"].to_numpy()
    predicted = predictions["ID3"].to_numpy()
    counts = np.array([[np.sum((actual == a) & (predicted == p)) for p in labels] for a in labels])
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, counts.max()))
    for (row, column), value in np.ndenumerate(counts):
        ax.text(column, row, str(value), ha="center", va="center", fontsize=16,
                color="white" if value > counts.max() / 2 else "#1e293b")
    ax.set(xticks=range(2), xticklabels=labels, yticks=range(2), yticklabels=labels,
           xlabel="Predicted label", ylabel="Actual label", title="Held-out predictions: synthetic example")
    return fig
