"""Main entry point: train four YOLOv3 comparisons and export aggregate results."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd
import torch

from .data import load_data, estimate_anchors
from .model import SqueezeYOLOv3
from .prepare_data import DEFAULT_DATA
from .training import train_model, evaluate

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
SEED = 42
EXPERIMENTS = [
    {"name": "Baseline", "learning_rate": 0.001, "enhanced": False, "freeze": False},
    {"name": "Lower learning rate", "learning_rate": 0.0005, "enhanced": False, "freeze": False},
    {"name": "Enhanced augmentation", "learning_rate": 0.001, "enhanced": True, "freeze": False},
    {"name": "Augmentation + frozen stem", "learning_rate": 0.001, "enhanced": True, "freeze": True},
]


def run_experiment(data_root=DEFAULT_DATA, epochs=30, threads=2):
    if epochs < 2 or threads < 1:
        raise ValueError("Use at least two epochs and one CPU thread.")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    data = load_data(data_root, SEED)
    anchors = estimate_anchors(data, seed=SEED)
    models, rows, history = {}, [], []
    for config in EXPERIMENTS:
        torch.manual_seed(SEED)
        model = SqueezeYOLOv3(anchors)
        fitted = train_model(model, data, config, epochs=epochs, seed=SEED)
        models[config["name"]] = model
        history.extend({"model": config["name"], **row} for row in fitted["history"])
        rows.append({"model": config["name"], "best_epoch": fitted["best_epoch"],
                     "validation_ap50": fitted["validation_ap50"], "training_seconds": fitted["seconds"]})
    # Fix the comparison and winning configuration before evaluating the test set.
    selected = max(rows, key=lambda row: row["validation_ap50"])["model"]
    curves = []
    for row in rows:
        result = evaluate(models[row["model"]], data, data["test"])
        row.update({"test_ap50": result["ap50"], "test_precision_at_0_5": result["precision"],
                    "test_recall_at_0_5": result["recall"], "selected_by_validation": row["model"] == selected})
        # Only aggregate PR points, no per-image detections or target coordinates.
        curves.extend({"model": row["model"], "recall": float(recall), "precision": float(precision)}
                      for recall, precision in zip(result["curve"]["recall"], result["curve"]["precision"]))
    metadata = {
        "data_source": "Locally supplied vehicle images and bounding boxes; not redistributed",
        "source_url": "https://www.mathworks.com/help/vision/ug/object-detection-using-yolo-v3-deep-learning.html",
        "data_sha256": data["data_sha256"], "images": len(data["images"]),
        "objects": sum(len(boxes) for boxes in data["boxes"]), "unique_pixel_groups": len(set(data["groups"])),
        "split_counts": {key: len(data[key]) for key in ("train", "validation", "test")},
        "split_object_counts": {key: sum(len(data["boxes"][i]) for i in data[key])
                                 for key in ("train", "validation", "test")},
        "split_rule": "GroupShuffleSplit by exact RGB pixel hash; approx 60/20/20; seed 42",
        "seed": SEED, "device": "cpu", "torch_threads": threads, "epochs": epochs, "batch_size": 8,
        "model": "SqueezeNet 1.1 + two YOLOv3 heads, 14x14 and 28x28; one vehicle class",
        "parameters": sum(p.numel() for p in model.parameters()), "input_size": [3, 227, 227],
        "initialization": "from scratch, no pretrained weights", "anchors": anchors.tolist(),
        "anchor_fit": "six width-height IoU-distance median clusters, training boxes only",
        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 0.0005,
        "gradient_clip_norm": 10, "warmup_steps": fitted["warmup_steps"],
        "freeze_stem_epoch": epochs // 2 + 1, "experiments": EXPERIMENTS,
        "selection_rule": "highest validation AP50; earliest epoch/configuration on ties",
        "selected_model": selected, "ap": "all-points interpolated AP at IoU 0.5",
        "nms_iou": 0.5, "ap_score_floor": 0.001, "max_detections_per_image": 100,
        "precision_recall_score_cutoff": 0.5,
        "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "pillow")},
    }
    return {"metrics": pd.DataFrame(rows), "history": pd.DataFrame(history),
            "pr_curve": pd.DataFrame(curves, columns=["model", "recall", "precision"]), "metadata": metadata}


def save_results(result, output=DEFAULT_OUTPUT):
    """Export aggregate tables/plots only: no pictures, weights, or annotations."""
    from .plotting import save_plots
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("metrics", "history", "pr_curve"):
        result[key].to_csv(output / f"{key}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    save_plots(result, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    result = run_experiment(args.data, args.epochs, args.threads)
    save_results(result, args.output)
    print(result["metrics"].round(4).to_string(index=False))
    print("Saved seven aggregate result files. No photos, annotations or weights were exported.")


if __name__ == "__main__":
    main()
