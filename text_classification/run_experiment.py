"""Main entry point: compare SMS baselines and recurrent classifiers, then save summaries."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import torch

from .data import DATA_FILE, SOURCE_URL, load_data, tokenize
from .models import SpamClassifier
from .training import classification_metrics, evaluate, train_model

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
SEED = 42
EXPERIMENTS = [
    {"name": "LSTM baseline", "learning_rate": 0.01, "head_sizes": [10], "bidirectional": False},
    {"name": "LSTM lower learning rate", "learning_rate": 0.001, "head_sizes": [10], "bidirectional": False},
    {"name": "LSTM deeper head", "learning_rate": 0.001, "head_sizes": [30, 10], "bidirectional": False},
    {"name": "BiLSTM deeper head", "learning_rate": 0.001, "head_sizes": [30, 10], "bidirectional": True},
]


def build_tfidf_baseline():
    """The pipeline fits both vocabulary/IDF and the classifier on training text only."""
    return make_pipeline(TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                         ngram_range=(1, 2), max_features=10000),
                         MultinomialNB(alpha=1.0))


def run_experiment(data_path=DATA_FILE, epochs=5, threads=1):
    if epochs < 1 or threads < 1:
        raise ValueError("Use positive epoch and CPU thread counts.")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    data = load_data(data_path, seed=SEED)
    texts = np.asarray(data["texts"], dtype=object)
    labels = data["labels"]
    train, validation, test = data["train"], data["validation"], data["test"]
    fitted_models, rows, history = {}, [], []

    # 1. Fit every predeclared baseline and network without inspecting test scores.
    baselines = [("Majority baseline", DummyClassifier(strategy="most_frequent")),
                 ("TF-IDF + naive Bayes", build_tfidf_baseline())]
    for name, model in baselines:
        started = perf_counter()
        # DummyClassifier ignores these features; the text pipeline consumes full messages.
        features = np.zeros((len(texts), 1)) if name == "Majority baseline" else texts
        model.fit(features[train], labels[train])
        valid = classification_metrics(labels[validation], model.predict(features[validation]))
        fitted_models[name] = (model, features)
        rows.append({"model": name, "best_epoch": 0, "parameters": 0,
                     "training_seconds": perf_counter() - started,
                     **{f"validation_{key}": value for key, value in valid.items()}})
    for config in EXPERIMENTS:
        print(config["name"], flush=True)
        torch.manual_seed(SEED)
        model = SpamClassifier(len(data["vocabulary"]), head_sizes=config["head_sizes"],
                                bidirectional=config["bidirectional"])
        fitted = train_model(model, data, config["learning_rate"], epochs=epochs, seed=SEED)
        valid = evaluate(model, data, validation)
        fitted_models[config["name"]] = model
        history.extend({"model": config["name"], **row} for row in fitted["history"])
        rows.append({"model": config["name"], "best_epoch": fitted["best_epoch"],
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "training_seconds": fitted["training_seconds"],
                     **{f"validation_{key}": valid[key] for key in ("accuracy", "spam_precision", "spam_recall", "spam_f1")}})

    # 2. Freeze the model choice; every model uses argmax, with no threshold search.
    selected = max(rows, key=lambda row: row["validation_spam_f1"])["model"]
    confusion_rows = []
    # 3. Evaluate all fixed comparisons once on the independent test split.
    for row in rows:
        name = row["model"]
        if name in dict(baselines):
            model, features = fitted_models[name]
            predictions = model.predict(features[test])
            scores = classification_metrics(labels[test], predictions)
            confusion = confusion_matrix(labels[test], predictions, labels=[0, 1])
        else:
            result = evaluate(fitted_models[name], data, test)
            scores = {key: result[key] for key in ("accuracy", "spam_precision", "spam_recall", "spam_f1")}
            confusion = result["confusion"]
        row.update({f"test_{key}": value for key, value in scores.items()})
        row["selected_by_validation"] = name == selected
        confusion_rows.extend({"model": name, "actual": actual, "predicted": predicted,
                               "count": int(confusion[i, j])}
                              for i, actual in enumerate(("ham", "spam"))
                              for j, predicted in enumerate(("ham", "spam")))

    token_lengths = [len(tokenize(texts[i])) for i in train]
    bins = [0, 10, 20, 30, 40, 50, 60, 100, max(200, max(token_lengths) + 1)]
    histogram, _ = np.histogram(token_lengths, bins=bins)
    metadata = {
        "data_source": "UCI SMS Spam Collection, official text version",
        "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository.",
        "source_url": "https://doi.org/10.24432/C5CC84", "download_url": SOURCE_URL,
        "data_license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "data_sha256": data["data_sha256"], "source_rows": data["source_rows"],
        "source_class_counts": data["source_class_counts"], "duplicates_removed": data["duplicates_removed"],
        "unique_messages": len(texts), "deduplication": "lowercase Unicode word tokens joined with spaces; keep first",
        "split_rule": "stratified 70/15/15 after normalized exact deduplication; fixed seed",
        "split_counts": {name: len(data[name]) for name in ("train", "validation", "test")},
        "class_counts": {name: {label: int((labels[data[name]] == code).sum()) for label, code in (("ham", 0), ("spam", 1))}
                         for name in ("train", "validation", "test")},
        "train_length_histogram": {"bin_edges": bins, "counts": histogram.tolist()},
        "train_messages_over_60_tokens": int(sum(length > 60 for length in token_lengths)),
        "seed": SEED, "device": "cpu", "torch_threads": threads, "epochs": epochs, "batch_size": 16,
        "embedding_size": 50, "hidden_size_per_direction": 80, "max_length": 60,
        "max_vocabulary_words": 10000, "vocabulary_size_including_reserved": len(data["vocabulary"]),
        "preprocessing": "train-only vocabulary; first 60 tokens; right padding; packed LSTM; empty/unknown -> UNK",
        "tfidf": {"max_features": 10000, "ngram_range": [1, 2], "alpha": 1.0,
                  "features_fitted": len(fitted_models["TF-IDF + naive Bayes"][0][0].vocabulary_)},
        "optimizer": "Adam", "gradient_clip_norm": 2.0, "pretrained_embeddings": False,
        "class_weighting": "none", "prediction_rule": "argmax (ham on an exact tie); no threshold tuning",
        "selection_rule": "maximum validation spam F1; earliest epoch/configuration on exact ties",
        "selected_model": selected, "experiments": EXPERIMENTS,
        "timing_note": "training_seconds includes fitting and validation/checkpoint selection",
        "packages": {name: version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "matplotlib")},
    }
    return {"metrics": pd.DataFrame(rows), "history": pd.DataFrame(history),
            "confusion": pd.DataFrame(confusion_rows), "metadata": metadata}


def save_results(result, output=DEFAULT_OUTPUT):
    """Export aggregate metrics and plots, never message text, vocabulary or weights."""
    from .plotting import save_plots
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("metrics", "history", "confusion"):
        result[key].to_csv(output / f"{key}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    save_plots(result, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    result = run_experiment(args.data, args.epochs, args.threads)
    save_results(result, args.output)
    columns = ["model", "best_epoch", "validation_spam_f1", "test_accuracy", "test_spam_precision", "test_spam_recall", "test_spam_f1"]
    print(result["metrics"][columns].round(4).to_string(index=False))
    print(f"Selected by validation spam F1: {result['metadata']['selected_model']}")
    print("Saved eight aggregate result files; no SMS text, vocabulary, or model weights were exported.")


if __name__ == "__main__":
    main()
