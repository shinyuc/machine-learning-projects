# Machine Learning Projects in Python

Python implementations and reproducible experiments exploring machine learning
algorithms, optimization, and model evaluation. Each module separates reusable
Python code from a Jupyter notebook that explains the problem, experiments,
and results.

| Module | Topic | Status |
| --- | --- | --- |
| [Linear regression](linear_regression/README.md) | Least squares, mini-batch GD, SGD | Implemented |
| [Decision trees](decision_trees/README.md) | Handwritten ID3, entropy, information gain, categorical prediction | Implemented |
| [Vehicle classification](svm_vehicle_classification/README.md) | Handwritten HOG, scikit-learn linear/RBF SVC | Implemented |
| [Neural network classification](neural_network_classification/README.md) | PyTorch MLP/CNN, activation and BatchNorm comparisons | Implemented |
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

## Run the decision-tree project

With the same environment activated, the main entry point is
[decision_trees/run_experiment.py](decision_trees/run_experiment.py):

```bash
python -m decision_trees.run_experiment
```

It generates a small synthetic dataset, learns a categorical ID3 tree, compares
it with a one-split tree and a majority baseline, and saves all seven result
files under [decision_trees/results/](decision_trees/results/). No dataset
download is needed. See the [module README](decision_trees/README.md) for the
output list and [the executed notebook](decision_trees/analysis.ipynb) for a
step-by-step explanation. To rerun every notebook output:

```bash
python -m jupyterlab decision_trees/analysis.ipynb
```

Select the `.venv` kernel, run all cells from a restarted kernel, and save.

## Run the SVM image-classification project

With the same environment activated, update the editable installation to
include the new module, then run its main entry point,
[svm_vehicle_classification/run_experiment.py](svm_vehicle_classification/run_experiment.py):

```bash
python -m pip install -e '.[notebook,dev]'
python -m svm_vehicle_classification.run_experiment
```

This generates teaching images, extracts HOG features, trains scikit-learn
SVC models with linear/RBF kernels, and saves all six result
files under [svm_vehicle_classification/results/](svm_vehicle_classification/results/).
No dataset download is needed. The default results describe generated shapes,
not real-world vehicle recognition. See the [module README](svm_vehicle_classification/README.md)
for the output list and optional local-image input.

```bash
python -m jupyterlab svm_vehicle_classification/analysis.ipynb
```

The [executed notebook](svm_vehicle_classification/analysis.ipynb) explains HOG,
kernel choices, support vectors, and the limits of the demonstration.

## Run the neural-network project

Install the PyTorch extra in the shared environment, then run the main entry
point [neural_network_classification/run_experiment.py](neural_network_classification/run_experiment.py):

```bash
python -m pip install -e '.[neural,notebook,dev]'
python -m neural_network_classification.run_experiment
```

This trains five MLP/CNN configurations, chooses models using validation loss,
evaluates a separate test set, and writes all eight files under
[neural_network_classification/results/](neural_network_classification/results/).
It runs on the CPU using scikit-learn's locally installed UCI digit example;
no separate dataset download is needed. These are 8 × 8 images resized to
28 × 28, not MNIST. No raw image files or model weights are exported.

```bash
python -m jupyterlab neural_network_classification/analysis.ipynb
```

The [module README](neural_network_classification/README.md) and
[executed notebook](neural_network_classification/analysis.ipynb) explain the
architectures, training loop, comparisons, and limitations.

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
decision_trees/
  tree.py             # Small ID3 functions using dictionaries and recursion
  run_experiment.py   # Main entry point: generate, train, evaluate, save
  plotting.py         # Tree and confusion-matrix figures
  analysis.ipynb      # Executed explanation and experiment walkthrough
  results/            # Seven generated result files; no external dataset
svm_vehicle_classification/
  features.py         # NumPy HOG descriptor
  data.py             # Generated images and optional local crops
  run_experiment.py   # Main entry point: generate, fit SVC, evaluate, save
  plotting.py         # Image, feature, and confusion-matrix figures
  analysis.ipynb      # Executed explanation and experiment walkthrough
  results/            # Six generated-example result files
neural_network_classification/
  models.py           # MLP and LeNet-style CNN layer definitions
  training.py         # Mini-batch training, validation selection, evaluation
  data.py             # Locally installed digits and fixed train/validation/test split
  run_experiment.py   # Main entry point: train five configurations, compare, save
  plotting.py         # Learning curves, metrics, class counts, confusion matrix
  analysis.ipynb      # Executed explanation and experiment walkthrough
  results/            # Eight result files; no raw images or weights
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

This project's own copyrightable contributions are released under the
[MIT License](LICENSE). This does not relicense third-party code, publications,
datasets, or separately installed dependencies.
See [Sources and third-party material](THIRD_PARTY_NOTICES.md) and each module's
references for implementation history, dataset attribution, and library
documentation, including the history of the removed handwritten SVM solver.
