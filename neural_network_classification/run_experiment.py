"""Main program: compare five small neural networks on handwritten digits."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

from .data import load_data
from .models import build_mlp, build_cnn
from .training import train_model, evaluate

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
SEED = 42
EXPERIMENTS = [
    {"name": "MLP sigmoid", "architecture": "mlp", "activation": "sigmoid", "batch_norm": False, "learning_rate": 0.01},
    {"name": "MLP ReLU", "architecture": "mlp", "activation": "relu", "batch_norm": False, "learning_rate": 0.01},
    {"name": "MLP ReLU + BN", "architecture": "mlp", "activation": "relu", "batch_norm": True, "learning_rate": 0.01},
    {"name": "MLP ReLU + BN (lr=0.1)", "architecture": "mlp", "activation": "relu", "batch_norm": True, "learning_rate": 0.1},
    {"name": "CNN", "architecture": "cnn", "activation": "relu", "batch_norm": True, "learning_rate": 0.01},
]


def run_experiment(epochs=20):
    """Fixed CPU experiment. Select by validation loss before any test evaluation.

    One CPU thread and deterministic operations make this small run repeatable
    in the same environment; results can differ across platforms/versions.
    """
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_data(SEED)
    images, labels = data["images"], data["labels"]
    train, validation, test = data["train"], data["validation"], data["test"]
    models, rows, history = {}, [], []

    # 1. Fit every predeclared configuration using only train and validation data.
    for config in EXPERIMENTS:
        torch.manual_seed(SEED)
        model = (build_cnn() if config["architecture"] == "cnn" else
                 build_mlp(config["activation"], config["batch_norm"]))
        fitted = train_model(model, images[train], labels[train], images[validation], labels[validation],
                             epochs=epochs, learning_rate=config["learning_rate"], seed=SEED)
        name = config["name"]
        models[name] = fitted["model"]
        history.extend({"model": name, **row} for row in fitted["history"])
        training = evaluate(model, images[train], labels[train])
        valid = evaluate(model, images[validation], labels[validation])
        rows.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                     "best_epoch": fitted["best_epoch"], "train_accuracy": training["accuracy"],
                     "validation_loss": valid["loss"], "validation_accuracy": valid["accuracy"]})

    # 2. Freeze the choice before inspecting test labels or test scores.
    selected = min(rows, key=lambda row: row["validation_loss"])["model"]

    # 3. Report all fixed comparisons, without changing the selected model.
    predictions = pd.DataFrame({"row_id": test, "actual": labels[test].numpy()})
    for row in rows:
        evaluation = evaluate(models[row["model"]], images[test], labels[test])
        predictions[row["model"]] = evaluation["predictions"]
        row.update(test_loss=evaluation["loss"], test_accuracy=evaluation["accuracy"],
                   test_macro_f1=f1_score(labels[test].numpy(), evaluation["predictions"], average="macro"),
                   selected_by_validation=row["model"] == selected)

    # A training-majority classifier provides a simple non-neural reference.
    majority = int(torch.bincount(labels[train]).argmax())
    metadata = {
        "data_source": "UCI optical digits via scikit-learn load_digits (not MNIST)",
        "source_url": "https://doi.org/10.24432/C50P49", "data_license": "CC BY 4.0",
        "data_sha256": data["data_sha256"], "original_image_shape": [8, 8],
        "model_input_shape": [1, 28, 28], "preprocessing": "divide by 16; bilinear resize, align_corners=False",
        "seed": SEED, "device": "cpu", "torch_threads": 1, "epochs": epochs,
        "batch_size": 64, "optimizer": "SGD", "momentum": 0.9, "experiments": EXPERIMENTS,
        "split_row_ids": {name: data[name].tolist() for name in ("train", "validation", "test")},
        "split_counts": {name: len(data[name]) for name in ("train", "validation", "test")},
        "selection_rule": "lowest validation cross-entropy; first encountered on exact ties",
        "selected_model": selected,
        "majority_baseline": {"class": majority, "test_accuracy": float(np.mean(labels[test].numpy() == majority))},
        "packages": {name: version(name) for name in ("torch", "numpy", "pandas", "matplotlib", "scikit-learn")},
    }
    return {"data": data, "models": models, "metrics": pd.DataFrame(rows),
            "history": pd.DataFrame(history), "predictions": predictions, "metadata": metadata}


def save_results(result, output=DEFAULT_OUTPUT):
    """Write eight summaries/figures, without raw images, pixel arrays, or weights."""
    import matplotlib.pyplot as plt
    from .plotting import plot_learning_curves, plot_accuracy, plot_confusion, plot_class_counts

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("metrics", "history", "predictions"):
        result[key].to_csv(output / f"{key}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    figures = {"learning_curves": plot_learning_curves(result), "accuracy_comparison": plot_accuracy(result),
               "confusion_matrix": plot_confusion(result), "class_counts": plot_class_counts(result)}
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)
    return 4 + len(figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    result = run_experiment(args.epochs)
    print(result["metrics"].round(4).to_string(index=False))
    print(f"Selected by validation loss: {result['metadata']['selected_model']}")
    count = save_results(result, args.output)
    print(f"Saved {count} result files to {args.output}")


if __name__ == "__main__":
    main()
