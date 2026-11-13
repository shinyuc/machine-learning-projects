# Machine Learning Projects in Python

Python implementations and reproducible experiments exploring machine learning
algorithms, optimization, and model evaluation. Each module separates reusable
Python code from a Jupyter notebook that explains the problem, experiments,
and results.

| Module | Topic | Status |
| --- | --- | --- |
| [Linear regression](linear_regression/README.md) | Least squares, mini-batch GD, SGD | Implemented |
| [Decision trees](decision_trees/README.md) | Decision trees | Planned |
| [Vehicle classification](svm_vehicle_classification/README.md) | HOG features and kernel SVM | Planned |
| [Neural network classification](neural_network_classification/README.md) | MLP and CNN image classification | Planned |
| [Object detection](object_detection/README.md) | Vehicle object detection | Planned |
| [Text classification](text_classification/README.md) | LSTM and BiLSTM text classification | Planned |

## Start with linear regression

Run these commands from this `machine-learning-projects` repository root,
not from the enclosing local workspace. Python 3.12 is the tested version.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook,dev]'
python -m linear_regression.data --download
```

The **main entry point** for the implemented linear-regression project is
[linear_regression/run_experiment.py](linear_regression/run_experiment.py).
With the environment activated and data downloaded, run it as a module from
the repository root:

```bash
python -m linear_regression.run_experiment
```

This one command fits and evaluates all comparison models, runs the
learning-rate and timing experiments, and saves all nine experiment artifacts
to [linear_regression/results/](linear_regression/results/): four CSV files,
four PNG figures, and one JSON metadata file. It also prints the evaluation
metrics in the terminal. Running it again replaces those result files;
measured runtimes can change. See the [module README](linear_regression/README.md#results)
for the complete output list. This command runs without opening Jupyter.

To reproduce the complete notebook, including its additional data exploration
and full-data numerical reference, open it and select **Restart Kernel and Run
All Cells**:

```bash
python -m jupyterlab linear_regression/analysis.ipynb
```

In VS Code, select the project's `.venv` interpreter for both Python files and
the notebook kernel. The notebook contains saved outputs, so its existing
results can also be read without running training again.

Run the correctness checks separately with `python -m pytest`.

## Repository layout

```text
linear_regression/
  algorithms.py       # NumPy least squares and handwritten gradient updates
  data.py             # Data acquisition, validation, splitting, scaling
  experiments.py      # Shared experiment used by the CLI and notebook
  plotting.py         # Shared figures
  run_experiment.py   # Command-line entry point and artifact export
  analysis.ipynb      # Executed explanation and experiment walkthrough
  results/            # Small metrics, coefficients, plots and run metadata
decision_trees/       # Planned
svm_vehicle_classification/  # Planned
neural_network_classification/  # Planned
object_detection/     # Planned
text_classification/  # Planned
tests/                # Numerical correctness and data-boundary tests
pyproject.toml        # Dependencies and package configuration
data/                 # Local downloaded data, ignored by Git
```

Only source code, notebooks, documentation, tests, and curated experimental
outputs belong in this repository. `.gitignore` excludes local datasets,
environments, checkpoints, and scratch work. Dataset acquisition and
attribution are documented in each module.

## Reproducibility

The default linear-regression experiment uses a fixed split and random seed. Its output
metadata records the dataset SHA-256, row indices, hyperparameters, and package
versions. Timing is measured on the executing machine and will vary.
`requirements-lock.txt` records the complete Python 3.12 environment used for
the checked-in run. To recreate it in a fresh Python 3.12 environment:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

## License and sources

The original code and documentation in this repository are released under the
[MIT License](LICENSE). Datasets and separately installed dependencies retain
their own terms; the MIT License does not grant rights to those materials.
See [Sources and third-party material](THIRD_PARTY_NOTICES.md) and each module's
references for dataset attribution and library documentation.
