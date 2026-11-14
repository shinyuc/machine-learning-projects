# Vehicle classification with HOG and SVC

Classify small grayscale image crops using a handwritten NumPy HOG descriptor
and scikit-learn's `SVC`. Compare linear and Gaussian (RBF) kernels on the same
features and held-out split. The SVM solver is provided by the library.

The default experiment draws its own vehicle-like silhouettes and other shapes.
No external images or dataset download are needed. This demonstrates an image
classification pipeline; it is not a benchmark on road photographs or an
object detector that locates vehicles in a larger scene.

## Run everything

The **main entry point** is [run_experiment.py](run_experiment.py). From the
repository root, with the shared environment activated:

```bash
python -m pip install -e '.[notebook,dev]'
python -m svm_vehicle_classification.run_experiment
```

The experiment generates images, extracts HOG features, trains two SVC models,
prints metrics and library fit status, and saves all **six result files** to
[results/](results/). Rerunning replaces those named files. It runs without Jupyter.

To view or rerun the explanation:

```bash
python -m jupyterlab svm_vehicle_classification/analysis.ipynb
```

In VS Code, open [analysis.ipynb](analysis.ipynb), choose the project's `.venv`
kernel, then **Restart Kernel and Run All Cells** and save. The checked-in
notebook contains its outputs and calls the same experiment as the CLI.

## What the code does

1. Generate 100 images per class, each 64 × 64 pixels. Use seed 42 and a
   stratified 80/20 split: 160 training images and 40 test images.
2. Extract **HOG** with NumPy: unsigned gradient directions, nine orientation
   bins per 8 × 8 cell, and overlapping 2 × 2-cell blocks with L2-Hys
   normalization. Each image becomes 1,764 numbers. Votes are interpolated
   between orientation bins but not spatial cells; this is a compact variant.
3. Call `SVC(...).fit(X_train, y_train)` for each kernel. Use `C=1`,
   `gamma=0.125` for RBF, and stopping tolerance `1e-6`. In the alternate
   Gaussian notation, `sigma=2` because `gamma=1/(2*sigma**2)`.
4. Predict held-out labels and compare accuracy, positive-class F1, and
   support-vector counts. Both models receive exactly the same training rows.

HOG uses no dataset-fitted parameters. Kernel settings are fixed before the
experiment; there is no test-set tuning. Scikit-learn uses LIBSVM internally.
The library's `fit_status_` is recorded in the metadata (`0` means successful
fitting), along with its iteration count. These are library diagnostics, not
an independently computed optimality check. Decision scores are not probabilities.

To fit an SVM on HOG arrays in your own experiment:

```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", C=1, gamma=0.125, tol=1e-6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Results

Default generated-image run:

| Model | Train accuracy | Test accuracy | Test F1 | Support vectors |
| --- | ---: | ---: | ---: | ---: |
| SVC linear | 100% | 100% | 1.00 | 23 |
| SVC rbf | 100% | 100% | 1.00 | 105 |

Both models correctly classify **40/40 generated test images**. These shapes
are deliberately easy to separate. Perfect scores on this demonstration do
not establish real-world vehicle recognition performance or show that RBF is
better than a linear kernel. The RBF model uses more support vectors here, so
more training examples contribute to each prediction.

| Saved file | Contents |
| --- | --- |
| `metrics.csv` | Train/test accuracy, test F1, support-vector counts for both models |
| `predictions.csv` | Test row IDs, generated labels, and predictions |
| `metadata.json` | Seed, split IDs, settings, data hash, library versions, fit status |
| `synthetic_examples.png` | Examples drawn by the generator |
| `feature_example.png` | A generated crop, gradient magnitude, and HOG vector |
| `confusion_matrices.png` | Test confusion matrices for both SVC models |

## Optional local images

The same code can read images you already have permission to use. Keep them
outside the repository or in its Git-ignored `data/` directory:

```text
data/vehicle-crops/
  vehicles/          # Nested folders are allowed
  non-vehicles/
```

```bash
python -m svm_vehicle_classification.run_experiment --data data/vehicle-crops --per-class 100
```

The loader samples a balanced subset, converts it to grayscale, resizes to
64 × 64, and skips exact duplicate normalized images before splitting. An
encountered duplicate with conflicting class labels raises an error. Similar
crops and frames from the same sequence can still cross the split. A serious
benchmark needs a source/sequence-aware split and separate validation data.

Local-image runs default to the Git-ignored `work/local-svm-results/` directory.
They write **four files**: metrics, predictions, metadata, and confusion matrices.
They do not export input-image illustrations, filenames, feature arrays, or
trained models. Public demonstration results use their separate default
directory. `--output` selects another directory; keep local-data outputs under
`work/`. No local dataset is bundled or downloaded by this module.

## Reading the code

| File | Purpose |
| --- | --- |
| [run_experiment.py](run_experiment.py) | Generate/load → HOG → fit SVC → evaluate → save |
| [features.py](features.py) | HOG gradients, histograms, and normalization |
| [data.py](data.py) | Generated shapes and optional local image loading |
| [plotting.py](plotting.py) | Image examples, HOG illustration, confusion matrices |
| [analysis.ipynb](analysis.ipynb) | Executed walkthrough and interpretation |

Run the focused checks from the repository root:

```bash
python -m pytest tests/test_svm_vehicle_classification.py
```

Tests check known HOG gradients and invariances, duplicate-image handling,
training-only fitting, disjoint splits, metric/prediction consistency, and
exports that omit local input-image illustrations. The removed handwritten
optimizer's numerical tests are no longer part of this module.

## Sources and implementation history

- Navneet Dalal and Bill Triggs, [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf), 2005.
- scikit-learn, [SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

The current version uses the separately installed scikit-learn solver. The
previous handwritten SMO implementation and its optimization outputs have
been removed from the current files because its earlier reference-code
permissions were not verified. Existing Git history is a separate matter;
see [source history and license scope](../THIRD_PARTY_NOTICES.md#svm-implementation-history).
