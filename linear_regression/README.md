# Linear regression and optimization

Predict sales from TV, radio, and newspaper advertising budgets. This project
compares a NumPy least-squares solution, handwritten
mini-batch gradient descent, handwritten stochastic gradient descent, and
scikit-learn's `LinearRegression`. A training-mean predictor provides a simple
baseline.

Start with [the executed notebook](analysis.ipynb) for formulas, experiments,
plots, and interpretation. Read [algorithms.py](algorithms.py) for the core
implementation. The notebook and CLI call the same experiment code.

## Data

The Advertising dataset contains 200 rows and three predictors. Download it
from the [ISL authors' resource page](https://www.statlearning.com/resources-second-edition):

```bash
python -m linear_regression.data --download
```

Alternatively, import a local copy of the dataset:

```bash
python -m linear_regression.data --from-file /path/to/Advertising.csv
```

Both commands create `data/advertising/Advertising.csv`, which Git ignores.
The loader accepts title-case and lowercase column names and ignores the
row-number column.
Dataset ownership and terms remain with the original source; this repository
provides retrieval instructions rather than redistributing the CSV.
The authors' resource page supplies the download but does not state an
explicit redistribution license for this CSV. The saved notebook shows the
schema, aggregate statistics, and original experiment figures; it does not
include a table of raw dataset rows. Attribution is provided for the data and
does not itself grant redistribution rights.

## Main entry point: reproduce the experiment results

The main file is **[run_experiment.py](run_experiment.py)**. It calls the shared
experiment in [experiments.py](experiments.py), which uses the core algorithms
in [algorithms.py](algorithms.py).

After installing the environment and acquiring the dataset as described
above, activate the environment and run this command from the repository root
(the directory containing `pyproject.toml`):

```bash
python -m linear_regression.run_experiment
```

This command fits and evaluates least squares, mini-batch GD, SGD, the
scikit-learn reference, and the mean baseline. It also runs the learning-rate
comparison and fit-time benchmarks, prints the evaluation metrics, and writes
all nine files listed under [Results](#results) to `linear_regression/results/`.
Use the module command above so the package imports resolve correctly.
No notebook session is needed to generate these files. Running the default
command again replaces them; fit-time measurements vary between runs.

To try a different configuration without replacing the checked-in results:

```bash
python -m linear_regression.run_experiment \
  --learning-rate 0.1 --epochs 500 --batch-size 20 \
  --output work/linear-regression-lr-0.1
```

To reproduce **every notebook output**, including the additional data
inspection, exploratory plots, and full-data numerical reference, open the
notebook and select **Restart Kernel and Run All Cells**:

```bash
python -m jupyterlab linear_regression/analysis.ipynb
```

The notebook runs the same main experiment and exports the same nine result
files in its final code cell. Its additional tables and exploratory plots
appear inside the notebook. Save it after execution to retain those outputs.

Run the correctness checks separately with `python -m pytest`.

## Experiment design

- Split the 200 observations into 160 training and 40 test rows with seed 42.
- Fit min-max scaling on training rows only; apply the same transform to test
  rows without clipping values outside the training range.
- Use all three features and an intercept for every fitted model.
- Use learning rate 0.01 and 500 epochs, with batches of 20 for mini-batch GD
  and one example for SGD.
- Compare learning rates 0.001, 0.01, and 0.1 using **training loss only**.
  These exploratory curves do not select a model using test scores.
- Report training half MSE, held-out RMSE, MAE, and R-squared.
- Measure median fit-only time over three repetitions after a warmup, with
  loss-history recording disabled. Loading, scaling, evaluation, and plotting
  are excluded. The main runs still record loss histories for visualization.

The test set is a single, small holdout. This is an optimizer comparison and
learning exercise, not evidence of performance across other populations or a
causal estimate of advertising effectiveness. Iterative solvers are compared
at a fixed epoch budget, not a matched convergence tolerance. One SGD epoch
has 160 updates; one mini-batch epoch has 8 updates.

## Implementation choices

1. The main experiment uses a held-out test set. A separate full-data fit in
   the notebook reports training error only and is excluded from the held-out
   comparison.
2. Least squares uses `numpy.linalg.lstsq` instead of explicitly forming
   `inv(X.T @ X)`. NumPy provides the numerical solver; the two gradient-based
   optimizers implement their updates directly without sklearn estimators.
3. Coefficients are converted back to original units with both slope and
   intercept corrections, with tests that verify predictions are preserved.
4. Loss is recorded at the same epoch boundaries for both iterative methods.
5. Partial final batches and constant-valued features are supported and tested.

For scaled inputs z = (x - minimum) / scale, the original-unit coefficients are
`coef = scaled_coef / scale` and
`intercept = scaled_intercept - minimum @ coef`.

## Results

The command-line entry point saves the following files. The notebook displays
and interprets the same main experiment results.

| Output file | Contents |
| --- | --- |
| [metrics.csv](results/metrics.csv) | Training loss, held-out RMSE, MAE, R-squared, and measured fit times |
| [coefficients.csv](results/coefficients.csv) | Model coefficients and intercepts in original feature units |
| [convergence.csv](results/convergence.csv) | Mini-batch GD and SGD training loss at each epoch |
| [learning_rates.csv](results/learning_rates.csv) | Training loss histories for learning rates 0.001, 0.01, and 0.1 |
| [metadata.json](results/metadata.json) | Dataset hash, split indices, settings, and environment versions |
| [convergence.png](results/convergence.png) | Training loss and distance to the least-squares optimum |
| [learning_rates.png](results/learning_rates.png) | Training convergence for each learning rate |
| [predictions.png](results/predictions.png) | Held-out predictions and residuals |
| [runtime.png](results/runtime.png) | Median fit-time comparison |

![Training convergence](results/convergence.png)

## References

- James, Witten, Hastie, and Tibshirani, *An Introduction to Statistical Learning*,
  [Advertising dataset](https://www.statlearning.com/resources-second-edition).
- [NumPy least-squares solver](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
- [scikit-learn guidance on preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
