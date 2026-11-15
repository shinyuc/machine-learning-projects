# SMS spam classification with LSTM and BiLSTM

Classify short English messages as `ham` (normal) or `spam`. This project
compares two simple baselines with four small recurrent networks. Reusable
Python files implement preprocessing, models, and training;
[analysis.ipynb](analysis.ipynb) explains the experiments and their limitations.

## Run all experiments

From the repository root, use the shared virtual environment:

```bash
source .venv/bin/activate
python -m pip install -e '.[neural,notebook,dev]'
python -m text_classification.data --download
python -m text_classification.run_experiment
```

The **main file is [run_experiment.py](run_experiment.py)**. The last command
trains/evaluates all six configurations and saves every result file. No Jupyter
session is required. The official dataset is downloaded once to Git-ignored
`data/sms-spam/`; subsequent runs are offline. Missing data produces an explicit
instruction to run the download command. The downloader checks the corpus hash
and retains the original dataset readme locally.

The default is five epochs per neural configuration, batch size 16, one CPU
thread, and seed 42. The checked run took about half a minute to fit and validate
the models; speed varies by machine. Embeddings start randomly, with no external
pretrained vectors. A separate longer run can use:

```bash
python -m text_classification.run_experiment --epochs 10 --output work/sms-long-run
```

This does not change the checked-in results. Avoid repeatedly tuning against the
reported test set. `--data /path/to/SMSSpamCollection` supports an existing local
copy of the same checksummed official file, not arbitrary CSV formats.

## What is preserved from the earlier experiment?

The recurrent classifiers retain 50-dimensional word embeddings, 80 hidden
units per direction, a 60-token limit, Adam, gradient clipping at 2, and five
training epochs. The original comparisons involved a learning-rate change,
a deeper dense head, and a switch from LSTM to BiLSTM. One additional comparison
separates the learning-rate change from the head change.

The notebook also preserves the small sequence-model shape exercises:

- Classification: input `(batch, time, 20)` → LSTM hidden size 30 → 10 logits
  per time step. Predicted class IDs have shape `(batch, time)`.
- Regression: input `(batch, time, 40)` → LSTM hidden size 100 → one numeric
  output per time step.

These use generated tensors to illustrate shapes; they are not extra trained
SMS models. The actual SMS network predicts one label per message.

## Data and preprocessing

This version uses the official [UCI SMS Spam Collection](https://doi.org/10.24432/C5CC84),
listed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The archive contains 5,574 messages: 4,827 ham and 747 spam. The earlier local
CSV had 5,572 rows and format/content differences, so the Python run is not an
exact replay of that file. The factory-report example's license does not apply
to the SMS CSV. See [source notices](../THIRD_PARTY_NOTICES.md) for the review.

The preprocessing is deliberately small and explicit:

1. Lowercase and extract Unicode word/number tokens. Punctuation separates
   tokens; there is no stemming, stopword removal, or pretrained tokenizer.
2. Remove duplicate normalized messages before splitting. Conflicting labels
   for an identical normalized message cause an error rather than silently
   choosing one label. This removes 444 repeated rows, leaving 5,130 messages.
3. Stratify into **3,590 train / 770 validation / 770 test** messages with seed 42.
4. Fit the neural vocabulary only on training text: up to 10,000 words plus
   `<pad>` and `<unk>`. The recorded vocabulary contains 7,250 entries in total.
5. Keep each neural input's first 60 tokens and right-pad. Empty normalized
   messages receive one `<unk>` token. Packed sequences prevent padding from
   changing the LSTM state. The TF-IDF pipeline learns its own vocabulary/IDF
   only on training text and uses full messages.

No duplicate normalized message crosses splits. Similar spam templates can
still cross them; no sender/time grouping is available. The split is random,
not chronological, and this older corpus is not a current production-spam test.

## Models and evaluation

| Configuration | Learning rate | Dense hidden units | Direction |
| --- | --- | --- | --- |
| Majority baseline | — | — | Always predicts the training-majority class |
| TF-IDF + naive Bayes | — | — | Word unigrams/bigrams, up to 10,000 features; alpha 1 |
| LSTM baseline | 0.01 | 10 | Forward |
| LSTM lower learning rate | 0.001 | 10 | Forward |
| LSTM deeper head | 0.001 | 30, 10 | Forward |
| BiLSTM deeper head | 0.001 | 30, 10 | Forward + backward |

Every dense hidden layer uses ReLU; the last layer returns two logits.
Cross-entropy handles the softmax internally during training. For BiLSTM, the
classifier concatenates the final forward and backward hidden states. Reading
the last padded output would give the wrong representation.

Neural runs restart with seed 42 and the same shuffled-batch seed. The two shallow
LSTMs have identical initial weights; different architectures have different
parameters. There is no class weighting or threshold tuning. All models use
argmax for prediction (ham on exact ties).

Each neural checkpoint maximizes **validation spam F1**, keeping the earliest
on ties. The overall model is selected using the same validation metric before
any test scores are computed. All predeclared comparisons are then reported on
the test set; test results do not change the selection.

Accuracy alone hides class imbalance: always predicting ham obtains 87.79%
test accuracy but catches no spam. Spam precision measures how often spam
predictions are right; recall measures how much actual spam is caught. F1 is
their harmonic mean. False positives are normal messages incorrectly blocked;
false negatives are spam messages missed.

## Recorded results

| Model | Best epoch | Validation spam F1 | Test accuracy | Test spam F1 |
| --- | --- | --- | --- | --- |
| Majority baseline | — | 0.0000 | 0.8779 | 0.0000 |
| TF-IDF + naive Bayes | — | 0.8101 | 0.9623 | 0.8176 |
| LSTM baseline | 5 | **0.9333** | **0.9831** | **0.9312** |
| LSTM lower learning rate | 3 | 0.9071 | 0.9792 | 0.9111 |
| LSTM deeper head | 5 | 0.9143 | 0.9805 | 0.9143 |
| BiLSTM deeper head | 5 | 0.9050 | 0.9753 | 0.8950 |

The validation-selected baseline LSTM caught 88 of 94 test spam messages and
misclassified 7 of 676 normal messages as spam. Its spam precision was 92.63%
and recall 93.62%. BiLSTM did not improve this run and took longer than the
otherwise matching forward-only network. This is one split and one seed, not
evidence that LSTM universally beats BiLSTM. Lower learning rates may need a
longer schedule. The changed dataset, deduplication, padding semantics, and new
held-out test split prevent direct comparison to the earlier MATLAB scores.

## Outputs and notebook

The main command writes eight files under `results/`:

- `metrics.csv`: validation/test scores, selected epochs, and measured times.
- `history.csv`: neural training loss and validation metrics for each epoch.
- `confusion.csv`: aggregate test confusion counts for each model.
- `metadata.json`: attribution, data hash, aggregate counts, settings, versions.
- `learning_curves.png`, `metrics_comparison.png`, `confusion_matrices.png`,
  and `data_summary.png`.

No raw messages, phone numbers, corpus vocabulary, per-message predictions,
word clouds, or model weights are exported. Example sentences in the notebook
are invented; the actual dataset is not displayed. Dataset licensing and privacy
are distinct considerations, so data remains local even though UCI lists an
open license.

```bash
python -m jupyterlab text_classification/analysis.ipynb
```

Select the `.venv` kernel. By default the notebook reads saved aggregates and
recreates the analysis without downloading data or retraining. Set `RETRAIN =
True` in the experiment cell to call the same Python pipeline on local data and
replace the saved results. Running all cells includes both sequence-shape and
tokenization demonstrations.

## Code map and tests

| File | Responsibility |
| --- | --- |
| `data.py` | Official download, validation, deduplication, splits, vocabulary, encoding |
| `models.py` | Sequence shape demonstrations and the shared LSTM/BiLSTM classifier |
| `training.py` | Adam updates, padding-aware batches, validation checkpoint selection |
| `run_experiment.py` | Baselines, fixed comparisons, held-out evaluation and export |
| `plotting.py` | Aggregate plots, with no real-message display |

Run `python -m pytest`. Tests cover duplicate/split boundaries, train-only
vocabularies, unknown words and truncation, padding invariance, both LSTM
directions, checkpoint restoration, and class-imbalance metrics. They use
invented text, not a downloaded corpus. Exact timing and results may change
across package versions/platforms; the root lock file records the tested setup.

## Sources and rights

The earlier pipeline followed MathWorks'
[Classify Text Data Using Deep Learning](https://www.mathworks.com/help/deeplearning/ug/classify-text-data-using-deep-learning.html).
This version uses PyTorch/scikit-learn components and new experiment code;
original MATLAB helpers, example prose, and report figures are not distributed.
The new module's own contributions use MIT; installed libraries and data keep
their respective licenses. The repository's separate BSD-licensed YOLO file
retains its existing license.

Dataset attribution: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
Processing changes are described above. The downloader retains the archive's
original readme locally, including its copyright/disclaimer terms. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the source review and
license boundaries; this project does not claim a clean-room development process
or endorsement by the dataset authors.
