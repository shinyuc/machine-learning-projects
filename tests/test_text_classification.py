"""Correctness checks use invented text and tiny tensors, never private corpus excerpts."""

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from text_classification import data as data_module
from text_classification.data import build_vocabulary, encode_messages, prepare_messages, read_messages, tokenize
from text_classification.models import SequenceModel, SpamClassifier
from text_classification.run_experiment import build_tfidf_baseline
from text_classification import training


def fixture_messages():
    return pd.DataFrame([{"label": "ham" if i < 40 else "spam", "text": f"Invented item unique{i}"}
                         for i in range(80)])


def test_tokenization_vocabulary_unknowns_truncation_and_empty_text():
    assert tokenize("Café, ROOM_A! It's 12.") == ["café", "room_a", "it", "s", "12"]
    vocabulary = build_vocabulary(["pear apple pear", "banana"])
    assert vocabulary == {"<pad>": 0, "<unk>": 1, "pear": 2, "apple": 3, "banana": 4}
    before = vocabulary.copy()
    tokens, lengths = encode_messages(["Pear mystery apple banana", "!!!", "banana"], vocabulary, max_length=3)
    assert tokens.tolist() == [[2, 1, 3], [1, 0, 0], [4, 0, 0]]
    assert lengths.tolist() == [3, 1, 1]
    assert vocabulary == before


def test_deduplication_precedes_disjoint_repeatable_stratified_splits():
    frame = fixture_messages()
    frame.loc[len(frame)] = ["ham", "INVENTED, item unique0!!!"]
    data = prepare_messages(frame)
    assert data["duplicates_removed"] == 1 and len(data["texts"]) == 80
    splits = [set(data[key]) for key in ("train", "validation", "test")]
    assert set.union(*splits) == set(range(80))
    assert not splits[0] & splits[1] and not splits[0] & splits[2] and not splits[1] & splits[2]
    again = prepare_messages(frame)
    for key in ("train", "validation", "test"):
        assert np.array_equal(data[key], again[key])
        assert set(data["labels"][data[key]]) == {0, 1}


def test_conflicting_duplicate_labels_are_rejected():
    frame = fixture_messages()
    frame.loc[len(frame)] = ["spam", frame.text.iloc[0]]
    with pytest.raises(ValueError, match="conflicting labels"):
        prepare_messages(frame)


def test_load_data_does_not_learn_words_from_validation_or_test(tmp_path, monkeypatch):
    frame = fixture_messages()
    path = tmp_path / "fixture.txt"
    path.write_text("invented fixture")
    monkeypatch.setattr(data_module, "read_messages", lambda _: frame)
    data = data_module.load_data(path)
    for i in data["train"]:
        assert f"unique{i}" in data["vocabulary"]
    for i in np.r_[data["validation"], data["test"]]:
        assert f"unique{i}" not in data["vocabulary"]
        assert data["tokens"][i, 2] == 1


def test_text_reader_preserves_quotes_and_checks_bytes(tmp_path):
    path = tmp_path / "messages.txt"
    content = b'ham\tAn "invented"\tmessage\nspam\tAnother invented message\n'
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    frame = read_messages(path, expected_sha256=expected)
    assert len(frame) == 2 and frame.text.iloc[0] == 'An "invented"\tmessage'
    path.write_bytes(content + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_messages(path, expected_sha256=expected)


def test_download_reads_only_named_members_and_keeps_readme(tmp_path, monkeypatch):
    corpus = b"ham\tInvented fixture\n"
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("SMSSpamCollection", corpus)
        archive.writestr("readme", b"Synthetic license fixture")
        archive.writestr("../escape.txt", b"Never extract this")
    monkeypatch.setattr(data_module, "urlopen", lambda *a, **kw: BytesIO(buffer.getvalue()))
    monkeypatch.setattr(data_module, "SOURCE_SHA256", hashlib.sha256(corpus).hexdigest())
    path = data_module.download_data(tmp_path / "local")
    assert path.read_bytes() == corpus
    assert (path.parent / "original-readme.txt").read_bytes() == b"Synthetic license fixture"
    assert not (tmp_path / "escape.txt").exists()


def test_download_rejects_public_repository_folder():
    public_folder = Path(data_module.__file__).resolve().parent / "raw-messages"
    with pytest.raises(ValueError, match="keep the corpus"):
        data_module.download_data(public_folder)
    assert not public_folder.exists()


@pytest.mark.parametrize("bidirectional", [False, True])
def test_predictions_ignore_padding_and_preserve_unsorted_batch_order(bidirectional):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = SpamClassifier(12, embedding_size=4, hidden_size=5, head_sizes=(3,), bidirectional=bidirectional).eval()
    # The short sequence is first, so pack_padded_sequence must restore original order.
    tokens = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7]])
    lengths = torch.tensor([2, 4])
    with torch.no_grad():
        batched = model(tokens, lengths)
        separate = torch.cat([model(tokens[i:i + 1, :length], torch.tensor([length]))
                              for i, length in enumerate(lengths.tolist())])
        extra_padding = model(torch.nn.functional.pad(tokens, (0, 5)), lengths)
    assert torch.allclose(batched, separate, atol=1e-6)
    assert torch.allclose(batched, extra_padding, atol=1e-6)


def test_padding_embedding_has_zero_gradient_and_real_tokens_learn():
    torch.manual_seed(8)
    model = SpamClassifier(8, embedding_size=4, hidden_size=5, head_sizes=(), bidirectional=True)
    tokens = torch.tensor([[2, 3, 0], [4, 0, 0]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.embedding.weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(tokens, torch.tensor([2, 1])), torch.tensor([0, 1]))
    loss.backward()
    assert model.embedding.weight.grad[0].eq(0).all()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert torch.equal(model.embedding.weight[0], before[0])
    assert not torch.equal(model.embedding.weight[2:5], before[2:5])


def test_sequence_classification_and_regression_shapes():
    classifier = SequenceModel(20, 30, 10)
    regressor = SequenceModel(40, 100, 1)
    logits = classifier(torch.zeros(2, 7, 20))
    assert logits.shape == (2, 7, 10)
    assert logits.argmax(dim=-1).shape == (2, 7)
    assert regressor(torch.zeros(2, 7, 40)).shape == (2, 7, 1)


def test_spam_metrics_do_not_reward_predicting_only_ham():
    scores = training.classification_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert scores["accuracy"] == 0.75 and scores["spam_f1"] == 0
    scores = training.classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert scores["spam_precision"] == pytest.approx(2 / 3)
    assert scores["spam_recall"] == 1 and scores["spam_f1"] == pytest.approx(0.8)


def test_tfidf_pipeline_does_not_fit_held_out_tokens():
    model = build_tfidf_baseline()
    model.fit(["team meeting", "project report", "reward offer", "prize offer"], [0, 0, 1, 1])
    vocabulary = model[0].vocabulary_.copy()
    model.predict(["unseenword offer"])
    assert model[0].vocabulary_ == vocabulary and "unseenword" not in vocabulary


def test_training_restores_validation_selected_epoch(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = SpamClassifier(6, embedding_size=3, hidden_size=4, head_sizes=())
    data = {"tokens": torch.tensor([[2, 0], [3, 0], [2, 0], [3, 0], [2, 0], [3, 0]]),
            "lengths": torch.ones(6, dtype=torch.long), "labels": np.array([0, 1, 0, 1, 0, 1]),
            "train": np.arange(4), "validation": np.arange(4, 6)}
    snapshots = []
    real_evaluate = training.evaluate

    def validation_schedule(current, dataset, indices):
        result = real_evaluate(current, dataset, indices)
        snapshots.append({key: value.clone() for key, value in current.state_dict().items()})
        result["spam_f1"] = [0.9, 0.4][len(snapshots) - 1]
        return result

    monkeypatch.setattr(training, "evaluate", validation_schedule)
    fitted = training.train_model(model, data, learning_rate=0.01, epochs=2, batch_size=2)
    assert fitted["best_epoch"] == 1
    assert all(torch.equal(value, snapshots[0][key]) for key, value in model.state_dict().items())
    assert any(not torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0])
