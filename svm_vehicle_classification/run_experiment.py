"""Main program: image crops -> HOG -> linear/RBF SVM -> evaluation."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .data import make_demo_images, load_local_images
from .features import extract_features

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
LOCAL_OUTPUT = Path(__file__).resolve().parents[1] / "work" / "local-svm-results"
SEED, C, GAMMA, TOL = 42, 1.0, 0.125, 1e-6


def run_experiment(data_dir=None, per_class=100):
    """Use fixed settings and an 80/20 stratified split; no test-set tuning."""
    # 1. Read or generate images. Each label is -1 (other) or +1 (vehicle).
    if data_dir is None:
        images, labels = make_demo_images(per_class, SEED)
    else:
        images, labels = load_local_images(data_dir, per_class, SEED)
    train, test = train_test_split(np.arange(len(labels)), test_size=0.2,
                                   random_state=SEED, stratify=labels)

    # 2. HOG is a fixed per-image transform: it fits no parameters on test data.
    features = extract_features(images)
    X_train, X_test = features[train], features[test]
    y_train, y_test = labels[train], labels[test]

    # 3. Let scikit-learn train one SVM per kernel on the training rows only.
    models, scores, fits = {}, [], []
    predictions = pd.DataFrame({"row_id": test, "actual": y_test})
    for kernel in ("linear", "rbf"):
        model = SVC(C=C, kernel=kernel, gamma=GAMMA, tol=TOL).fit(X_train, y_train)
        models[kernel] = model
        predicted = model.predict(X_test)
        method = f"SVC {kernel}"
        predictions[method] = predicted
        scores.append({"method": method,
                       "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
                       "test_accuracy": accuracy_score(y_test, predicted),
                       "test_f1": f1_score(y_test, predicted, pos_label=1, zero_division=0),
                       "support_vectors": int(model.n_support_.sum())})
        fits.append({"kernel": kernel, "fit_status": int(model.fit_status_),
                     "iterations": int(model.n_iter_.sum())})
    metadata = {
        "data_source": "generated silhouettes" if data_dir is None else "local image folders",
        "seed": SEED, "images_per_class": per_class, "n_train": len(train), "n_test": len(test),
        "image_shape": [64, 64], "hog_features": features.shape[1],
        "hog": {"cell_size": 8, "orientation_bins": 9, "block_cells": [2, 2], "normalization": "L2-Hys"},
        "C": C, "gamma": GAMMA, "sigma_equivalent": float(np.sqrt(1 / (2 * GAMMA))),
        "estimator": "sklearn.svm.SVC", "solver_backend": "LIBSVM", "solver_tolerance": TOL,
        "train_row_ids": train.tolist(), "test_row_ids": test.tolist(),
        "data_sha256": hashlib.sha256(images.tobytes() + labels.tobytes()).hexdigest(),
        "fits": fits,
        "packages": {name: version(name) for name in ("numpy", "pandas", "matplotlib", "scikit-learn", "pillow")},
    }
    return {"images": images, "labels": labels, "features": features, "train": train, "test": test,
            "models": models, "metrics": pd.DataFrame(scores), "predictions": predictions,
            "metadata": metadata}


def save_results(result, output=DEFAULT_OUTPUT):
    """Export metrics and plots; image illustrations are only from the generator."""
    import matplotlib.pyplot as plt
    from .plotting import plot_confusion, plot_examples, plot_features

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("metrics", "predictions"):
        result[key].to_csv(output / f"{key}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    figures = {"confusion_matrices": plot_confusion(result)}
    if result["metadata"]["data_source"] == "generated silhouettes":
        figures.update(synthetic_examples=plot_examples(result), feature_example=plot_features(result))
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)
    return 3 + len(figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local root containing vehicles/ and non-vehicles/.")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    result = run_experiment(args.data, args.per_class)
    output = args.output or (LOCAL_OUTPUT if args.data else DEFAULT_OUTPUT)
    count = save_results(result, output)
    print(result["metrics"].round(4).to_string(index=False))
    for fit in result["metadata"]["fits"]:
        print(f"{fit['kernel']}: fit_status={fit['fit_status']} (0=success), "
              f"iterations={fit['iterations']}")
    print(f"Saved {count} result files to {output}")


if __name__ == "__main__":
    main()
