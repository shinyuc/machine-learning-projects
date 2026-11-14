# Neural network image classification

Compare a multilayer perceptron (MLP) and a small convolutional network (CNN)
on handwritten digits. The PyTorch code defines the networks, mini-batch
training loop, validation-based checkpoint selection, and held-out evaluation.
PyTorch supplies automatic differentiation and the SGD optimizer.

The experiment changes activation, Batch Normalization, and learning rate in
separate steps. It uses ordinary functions and `nn.Sequential` networks so
the layer order and training process are easy to follow.

## Run everything

The **main entry point** is [run_experiment.py](run_experiment.py). From the
repository root, activate the shared environment and install the neural extra:

```bash
source .venv/bin/activate
python -m pip install -e '.[neural,notebook,dev]'
python -m neural_network_classification.run_experiment
```

This one experiment command loads the digit example, trains all five
configurations for 20 epochs on the CPU, chooses checkpoints and a model by
validation loss, evaluates the test set, and saves all eight files under
[results/](results/). It runs without Jupyter or a GPU. Rerunning replaces
those result files. A shorter smoke run can use `--epochs 2 --output work/nn-smoke`;
its scores should not be compared with the saved 20-epoch results.

To read or rerun the explanation:

```bash
python -m jupyterlab neural_network_classification/analysis.ipynb
```

In VS Code, open [analysis.ipynb](analysis.ipynb), select the `.venv` kernel,
restart, run all cells, and save. The notebook already contains its outputs
and calls the same experiment functions as the command-line program.

## Data and evaluation protocol

We use the **1,797 8 × 8 digit images** supplied by scikit-learn's
[`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
They are a subset of the UCI Optical Recognition of Handwritten Digits data,
not MNIST. Once dependencies are installed, no additional download is needed.

The [UCI dataset page](https://doi.org/10.24432/C50P49) attributes the data to
E. Alpaydin and C. Kaynak (1998) and lists a
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
The raw data files are supplied by the separately installed library; they
are not copied into this repository. Saved outputs contain aggregate plots,
metrics, split IDs, and predictions, without source images or pixel arrays.

Preprocessing divides intensities by their documented maximum of 16 and
bilinearly resizes each image to 28 × 28 (`align_corners=False`). Resizing
matches the network input size but **does not add image detail**. Neither
operation fits parameters on the dataset.

| Split | Images | Purpose |
| --- | ---: | --- |
| Training | 1,077 | Gradient updates and BatchNorm running statistics |
| Validation | 360 | Select an epoch for each model and select the final model |
| Test | 360 | Evaluate the frozen choices |

All splits are stratified, disjoint, and use seed 42. This is a new random
split of the subset distributed by scikit-learn, not the original UCI benchmark
split. Writer identities are not supplied by this loader, so writer-independent
generalization is not established.

## Models and training

The MLP flattens each image and uses hidden layers of 120 and 84 units, followed
by ten class logits. All MLP variants start with identical dense-layer weights
and receive the same shuffled training batches.

| Configuration | Hidden activation | BatchNorm | Learning rate | Parameters |
| --- | --- | --- | ---: | ---: |
| MLP sigmoid | Sigmoid | No | 0.01 | 105,214 |
| MLP ReLU | ReLU | No | 0.01 | 105,214 |
| MLP ReLU + BN | ReLU | Yes | 0.01 | 105,622 |
| MLP ReLU + BN (lr=0.1) | ReLU | Yes | 0.1 | 105,622 |
| CNN | ReLU | Yes | 0.01 | 61,750 |

The CNN follows a **LeNet-style** pattern:

```text
1 × 28 × 28
→ Conv 5×5, 6 channels, padding 2 → BatchNorm → ReLU → MaxPool 2
→ Conv 5×5, 16 channels          → BatchNorm → ReLU → MaxPool 2
→ Flatten (16 × 5 × 5)
→ Linear 120 → ReLU → Linear 84 → ReLU → Linear 10
```

It uses modern activations, max pooling, and BatchNorm; it is not an exact
historical LeNet-5 reproduction.

Every configuration uses batch size 64, cross-entropy loss, and SGD with
momentum 0.9. Networks return **logits**: `CrossEntropyLoss` handles the
log-softmax calculation internally. No separate Softmax layer is used for
training.

The training loop is `zero_grad → forward → loss → backward → step`.
Full training and validation metrics are measured in evaluation mode after
each epoch, including epoch zero. Evaluation disables gradients and freezes
BatchNorm statistics. The lowest-validation-loss state is kept in memory,
including BatchNorm buffers, and restored after training. No model checkpoint
file is exported.

The overall model is selected by **lowest validation cross-entropy**, before
any test evaluation. All five fixed test comparisons are then reported. Test
scores do not influence the selected configuration or its epoch.

## Results

Saved 20-epoch run, with each model restored to its selected epoch:

| Model | Selected epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: |
| MLP sigmoid | 20 | 20.56% | 21.39% |
| MLP ReLU | 19 | 95.00% | 94.44% |
| MLP ReLU + BN | 20 | 98.61% | 98.06% |
| MLP ReLU + BN (lr=0.1) | 20 | 98.33% | 98.06% |
| CNN | 20 | 99.44% | 97.50% |

The selected model is **MLP ReLU + BN (lr=0.1)**. Its validation loss is about
0.0388, slightly below the CNN's 0.0390, even though the CNN has higher
validation accuracy. Cross-entropy depends on confidence as well as the
winning class, so those rankings need not match.

The sigmoid baseline has low training accuracy too: it learns slowly under
this initialization, learning rate, and fixed budget. This does not demonstrate
that sigmoid networks cannot solve the problem. ReLU and BatchNorm help in
this run. Increasing the learning rate does not improve the final test accuracy
over the lower-rate BatchNorm model. The small differences among the stronger
models should not be interpreted as a general ranking from one split and seed.

## Saved files

| File | Contents |
| --- | --- |
| `metrics.csv` | Parameter counts, selected epochs, validation/test metrics, selected model |
| `history.csv` | Training and validation loss/accuracy for epochs 0–20 |
| `predictions.csv` | Test row IDs, actual digit labels, all five predictions |
| `metadata.json` | Data source/license/hash, split IDs, settings, baseline, package versions |
| `learning_curves.png` | Train/validation curves and selected epochs |
| `accuracy_comparison.png` | Validation/test accuracy and a training-majority baseline |
| `confusion_matrix.png` | Test confusion matrix of the validation-selected model |
| `class_counts.png` | Class counts in each split |

## Reading and testing the code

Start with [models.py](models.py) for the network definitions, then
[training.py](training.py) for the training loop. [data.py](data.py) handles
the fixed preprocessing and splits. [run_experiment.py](run_experiment.py)
connects the experiment and export; [plotting.py](plotting.py) creates figures.

```bash
python -m pytest tests/test_neural_network_classification.py
```

Tests verify gradient flow, learning a known boundary, evaluation without
BatchNorm updates, sample-weighted loss, checkpoint restoration, deterministic
training, and separation of test labels from fitting and model selection.
These tests are skipped when the optional PyTorch dependency is absent.

The checked-in run uses Python 3.12 and the versions in
[`requirements-lock.txt`](../requirements-lock.txt), including PyTorch. It
uses one CPU thread and deterministic operations. Matching seeds do not
guarantee identical results across library versions, devices, or platforms.

## References

- Alpaydin, E. and Kaynak, C. (1998), [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49), UCI Machine Learning Repository, CC BY 4.0.
- PyTorch, [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
- PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) and [BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
- PyTorch, [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- MathWorks, [Create Simple Deep Learning Neural Network for Classification](https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html), background for the earlier local prototype.
- LeCun, Bottou, Bengio, and Haffner, [Gradient Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h), 1998, architectural background for the LeNet-style CNN.

The experiment builds on an earlier local prototype that followed the
MathWorks tutorial. The PyTorch training program, dataset, and evaluation
setup are described above; MathWorks source files and example data are not
bundled. See [implementation history and dataset terms](../THIRD_PARTY_NOTICES.md#handwritten-digit-example).
