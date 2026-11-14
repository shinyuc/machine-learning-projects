"""Load the Advertising data and fit preprocessing on training rows only."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .algorithms import LinearModel, _features

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
DATA_URL = "https://www.statlearning.com/s/Advertising.csv"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "advertising" / "Advertising.csv"


def load_advertising(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Advertising data not found at {path}. "
            "Run `python -m linear_regression.data --download`, "
            "or use `--from-file /path/to/Advertising.csv`."
        )
    frame = pd.read_csv(path)
    # Ignore the original CSV's unnamed row-number column.
    required = FEATURES + [TARGET]
    canonical_names = {name.lower(): name for name in required}
    frame = frame.rename(columns={name: canonical_names.get(str(name).strip().lower(), name) for name in frame.columns})
    if not set(required).issubset(frame.columns):
        raise ValueError(f"CSV must contain the columns {required}.")
    frame = frame[required].apply(pd.to_numeric, errors="raise")
    if len(frame) < 2 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Data must contain at least two rows with no missing or infinite values.")
    return frame


def split_indices(n_samples: int, *, test_fraction: float = 0.2, seed: int = 42):
    """Return reproducible, disjoint training/test row indices."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("At least two samples are required.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")
    n_test = int(np.ceil(n_samples * test_fraction))
    if n_test >= n_samples:
        raise ValueError("The requested split leaves no training samples.")
    order = np.random.default_rng(seed).permutation(n_samples)
    return order[n_test:], order[:n_test]


@dataclass
class MinMaxScaler:
    minimum: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "MinMaxScaler":
        X = _features(X)
        minimum = X.min(axis=0)
        span = X.max(axis=0) - minimum
        return cls(minimum=minimum, scale=np.where(span == 0, 1.0, span))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _features(X)
        if X.shape[1] != len(self.minimum):
            raise ValueError("Feature count differs from the fitted scaler.")
        # Test values may fall outside [0, 1]; do not refit or clip them.
        return (X - self.minimum) / self.scale

    def to_original_scale(self, model: LinearModel) -> LinearModel:
        """Undo scaling for both slopes AND intercept without changing predictions."""
        if len(model.coef) != len(self.scale):
            raise ValueError("Model and scaler have different feature counts.")
        coef = model.coef / self.scale
        intercept = model.intercept - self.minimum @ coef
        return LinearModel(coef=coef, intercept=float(intercept))


def prepare_data(*, source: Path | None, destination: Path) -> Path:
    """Validate before replacing a local dataset; never commit the downloaded CSV."""
    destination = destination.resolve()
    if source is not None and source.resolve() == destination:
        load_advertising(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "Advertising.csv"
        if source is None:
            with urlopen(DATA_URL, timeout=30) as response, staged.open("wb") as output:
                shutil.copyfileobj(response, output)
        else:
            shutil.copyfile(source, staged)
        load_advertising(staged)
        staged.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="Download from the ISL authors' site.")
    mode.add_argument("--from-file", type=Path, help="Import an existing local Advertising CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    path = prepare_data(source=args.from_file, destination=args.output)
    print(f"Prepared {len(load_advertising(path))} rows at {path}")


if __name__ == "__main__":
    main()
