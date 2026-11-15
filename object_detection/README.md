# Vehicle detection with a small YOLOv3 model

Detect vehicles in real photographs using a SqueezeNet backbone, two YOLOv3
heads, six learned anchors, and non-maximum suppression (NMS). Core code is in
Python; [analysis.ipynb](analysis.ipynb) explains the experiment and displays
saved aggregate results without needing the photos.

This revisits an earlier local experiment with 227 × 227 inputs and two heads.
It is a **SqueezeNet-based YOLOv3 variant**, not the original Darknet-53,
three-scale network or an exact numerical reproduction of MATLAB.

## Run all experiments

From the repository root, activate the environment and install the extra:

```bash
source .venv/bin/activate
python -m pip install -e '.[detection,notebook,dev]'
```

The main file is **[run_experiment.py](run_experiment.py)**. Once the local data
has been prepared, this single command trains all four configurations,
selects checkpoints using validation AP, evaluates the test set, and saves all
seven result files:

```bash
python -m object_detection.run_experiment
```

The default is 30 epochs per configuration, batch size 8, and two CPU threads.
Training takes several minutes depending on the machine. Models start from
scratch; no pretrained weights are downloaded. The original longer training
budget is also available:

```bash
python -m object_detection.run_experiment --epochs 200
```

That replaces the saved results. For a separate exploratory run, choose a
Git-ignored output directory, for example `--output work/detection-long-run`.
No model weights are saved. The callable `detect()` in `yolo.py` performs
inference on a trained in-memory model.

## Prepare the local data once

Photographs and annotations are **not in this repository**. Supply an authorized
local copy of the vehicle example containing `vehicleImages/` and
`vehicleDatasetGroundTruth.mat`, then run:

```bash
python -m object_detection.prepare_data --source '/path/to/local/vehicle-example'
```

Replace the placeholder with your actual local directory. The converter reads
an older MATLAB table, copies images into `data/vehicle-detection/images/`, and
writes `data/vehicle-detection/annotations.json`. It converts 1-based
`[x, y, width, height]` boxes to 0-based `[x1, y1, x2, y2]` boxes with exclusive
right/bottom edges. It does not change source files. This output directory is
ignored by Git. Keep custom data directories outside Git too.

The converter uses SciPy's low-level reader for this particular older table
format, not a general MATLAB table loader. If your table format differs, supply
the same local JSON schema instead (these coordinates are invented):

```json
[{"file": "your_image.jpg", "boxes": [[10, 20, 80, 90]]}]
```

Place images in the sibling `images/` directory. There is one class, `vehicle`;
records may contain multiple boxes. Use `--data /path/to/local/data` for a
different root. At least 15 images and enough distinct image groups and training
boxes are needed for splitting and six anchors.

## Code layout

| File | Responsibility |
| --- | --- |
| `prepare_data.py` | Local table conversion and box validation |
| `data.py` | Fixed splits, duplicate grouping, training-only anchor clustering |
| `augmentation.py` | Flip, brightness, noise, erasing, box-aware crop |
| `model.py` | SqueezeNet features and two detection heads |
| `yolo.py` | Decode boxes, assign anchors, compute loss, apply NMS |
| `training.py` | SGD training and validation checkpoint selection |
| `evaluation.py` | Greedy object matching and interpolated AP50 |
| `run_experiment.py` | Complete comparison and summary export |
| `plotting.py` | Aggregate figures, with no photograph export |

RGB pixels are scaled to [0, 1] and resized directly to 227 × 227; box coordinates
are scaled with them. The `fire5` and `fire9` features feed 28 × 28 and 14 × 14
heads. Each cell predicts three anchors, each with four box values, objectness,
and one class probability. The final score is objectness times class probability.
NMS removes overlapping detections of the same object.

Six anchor widths/heights are estimated **only from training boxes** using
IoU-distance median clustering. Larger anchors go to the coarse head. Each true
object is assigned to its best anchor. Unassigned predictions with IoU ≥ 0.5 to
a real box are ignored in the background loss. A cell/anchor can represent one
object; on a collision, this simple implementation keeps the first assignment.

## Experiment design

| Configuration | Learning rate | Augmentation | Freeze |
| --- | --- | --- | --- |
| Baseline | 0.001 | Horizontal flip | None |
| Lower learning rate | 0.0005 | Horizontal flip | None |
| Enhanced augmentation | 0.001 | Flip, brightness, noise, erasing, crop | None |
| Augmentation + frozen stem | 0.001 | Same enhanced transforms | First convolution after halfway |

Erasing keeps the full object box to simulate occlusion. Cropping clips boxes
and discards objects with less than 50% visible area. Crop/resize adds zoom
variation; the baseline has no independent random-scale transform. This is an
explicit version of the earlier augmentation themes, with different strengths
and random draws.

Comparisons start from the same initialization and fixed seed. Training uses
SGD with momentum 0.9, weight decay 0.0005, a linear warmup over at most five
epochs (capped at 1,000 steps), and gradient norm clipping at 10. The first
convolution is genuinely frozen with `requires_grad_(False)` in the freeze
comparison. Validation may select a checkpoint from before freezing; inspect
`best_epoch` before interpreting its effect.

The supplied local data contains 295 images and 336 boxes, but only 246 unique
RGB pixel groups. Exact duplicates stay in the same split: **179 train,
56 validation, 60 test** images with seed 42. This approximates 60/20/20 by
groups rather than forcing exact image counts. Unlike the earlier 60/40 split,
this reserves validation data for choosing epochs/configurations. Related but
nonidentical scenes are not grouped because sequence identifiers are unavailable;
this remains a small demonstration, not a robust real-world benchmark.
Duplicate rows remain within their assigned split and retain their weight in
training and evaluation; the reported metrics are not deduplicated metrics.

Each model's checkpoint maximizes validation AP50 (earliest on ties). The best
configuration is fixed before any test evaluation. Test scores for all
predeclared comparisons are reported, without selecting a winner from them.

**AP50 is detection average precision at IoU 0.5**, not classification accuracy
or COCO mAP averaged over IoU thresholds. It integrates an interpolated
precision–recall curve, using score floor 0.001 and up to 100 detections per
image after NMS. Separate precision/recall columns use a fixed score cutoff
of 0.5. Low confidence can produce useful AP but poor recall at that cutoff.

## Saved results and notebook

The checked-in 30-epoch run produced:

| Configuration | Best epoch | Validation AP50 | Test AP50 |
| --- | --- | --- | --- |
| Baseline | 28 | 0.6907 | 0.6785 |
| Lower learning rate | 28 | 0.4693 | 0.4377 |
| Enhanced augmentation | 29 | 0.6786 | 0.6440 |
| Augmentation + frozen stem | 27 | 0.7004 | 0.5766 |

Validation selected the freeze configuration, but the baseline had higher
reported test AP50. Selection was not changed after seeing test results. The
small validation advantage did not carry over to this test split; this run
does not establish that freezing or stronger augmentation improves generalization.
The lower learning rate also learned less within this fixed training budget;
that does not show it would be worse with a longer schedule. Training all four
configurations took about ten minutes on the recorded CPU environment.

`results/` contains `metrics.csv`, `history.csv`, `pr_curve.csv`, `metadata.json`,
`learning_curves.png`, `precision_recall.png`, and `ap_comparison.png`.
Metadata records the data fingerprint, counts, settings, anchors, and package
versions. No images, annotation rows, per-image predictions, or weights are
exported. The main command overwrites these files.

```bash
python -m jupyterlab object_detection/analysis.ipynb
```

Select the `.venv` kernel and run all cells. By default the notebook reads saved
summaries and recreates the analysis, so readers do not need private images.
Set `RETRAIN = True` in its experiment cell to retrain from local data through
the same Python function. This uses the default settings and overwrites result
files. There is deliberately no photo-display cell, preventing an executed
photo output from accidentally entering Git.

Run correctness checks with `python -m pytest`. Detection tests use generated
fixtures, not the local dataset. Results can differ across PyTorch versions and
platforms; the checked environment is recorded in the root lock file.

## Sources and license

The adapted YOLO decoding/loss file uses Western Digital's BSD-3-Clause
[Python implementation](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch)
at the revision documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
Its full copyright/license notice is retained; the project's own remaining
contributions use MIT. PyTorch and torchvision are separately installed.

The earlier experiment and local data came from the MathWorks
[vehicle detection example](https://www.mathworks.com/help/vision/ug/object-detection-using-yolo-v3-deep-learning.html).
Many photographs originate from the Caltech Cars datasets. The example's
permission statement is not a redistribution grant to this project. Images,
annotations, original MATLAB helpers, and pretrained models are not bundled;
their original terms still apply to local use. See the root notices for the
implementation history and separate code/data license boundaries.
