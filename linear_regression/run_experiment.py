"""Run the regression comparison and save compact, reproducible artifacts."""

import argparse
import json
from pathlib import Path

from .data import DEFAULT_DATA_PATH
from .experiments import ExperimentConfig, ExperimentResult, run_experiment

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def save_results(result: ExperimentResult, output: str | Path = DEFAULT_OUTPUT):
    # Delay pyplot import so the CLI can choose a headless backend first.
    import matplotlib.pyplot as plt
    from .plotting import plot_convergence, plot_learning_rates, plot_predictions, plot_runtime

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(output / "metrics.csv")
    result.coefficients.to_csv(output / "coefficients.csv")
    result.histories.to_csv(output / "convergence.csv", index=False)
    result.learning_rates.to_csv(output / "learning_rates.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(result.metadata, indent=2) + "\n")
    for name, plot in [
        ("convergence", plot_convergence), ("learning_rates", plot_learning_rates),
        ("predictions", plot_predictions), ("runtime", plot_runtime),
    ]:
        figure = plot(result)
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    config = ExperimentConfig(
        seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, timing_repeats=args.timing_repeats,
    )
    result = run_experiment(args.data, config)
    save_results(result, args.output)
    print(result.metrics.round(5).to_string())
    print(f"\nSaved figures, metrics, coefficients and metadata to {args.output}")


if __name__ == "__main__":
    main()
