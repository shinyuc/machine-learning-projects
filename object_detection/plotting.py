"""Aggregate figures only; deliberately no photograph plotting/export function."""

import matplotlib.pyplot as plt


def learning_curves(result):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, rows in result["history"].groupby("model", sort=False):
        axes[0].plot(rows.epoch, rows.loss, label=name)
        axes[1].plot(rows.epoch, rows.validation_ap50, label=name)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="YOLOv3 training objective")
    axes[1].set(xlabel="Epoch", ylabel="Validation AP50", title="Checkpoint selection", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    return figure


def precision_recall(result):
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for name, rows in result["pr_curve"].groupby("model", sort=False):
        axis.step(rows.recall, rows.precision, where="post", label=name)
    axis.set(xlabel="Recall", ylabel="Precision", title="Test precision–recall at IoU 0.5",
             xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(fontsize=8)
    return figure


def comparison(result):
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    table = result["metrics"].set_index("model")[["validation_ap50", "test_ap50"]]
    table.plot.barh(ax=axis)
    axis.set(xlabel="AP50", ylabel="", xlim=(0, 1), title="Fixed comparison; selection uses validation only")
    axis.legend(["Validation", "Test"])
    return figure


def save_plots(result, output):
    for name, plot in (("learning_curves", learning_curves), ("precision_recall", precision_recall),
                       ("ap_comparison", comparison)):
        figure = plot(result)
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)
