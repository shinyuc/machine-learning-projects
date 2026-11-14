"""Checks for learning, evaluation modes, checkpoint selection, and data boundaries."""

from copy import deepcopy
from importlib import import_module

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install the project's neural extra to test PyTorch models.")
from torch import nn

from neural_network_classification.data import load_data
from neural_network_classification.models import build_mlp, build_cnn
from neural_network_classification.training import evaluate, train_model


@pytest.fixture(autouse=True)
def small_cpu_run():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous_threads)


def test_digit_splits_are_disjoint_stratified_and_repeatable():
    data, repeated = load_data(), load_data()
    assert data["images"].shape == (1797, 1, 28, 28)
    assert data["images"].dtype == torch.float32 and data["labels"].dtype == torch.long
    assert data["images"].min() >= 0 and data["images"].max() <= 1
    assert [len(data[name]) for name in ("train", "validation", "test")] == [1077, 360, 360]
    splits = [set(data[name]) for name in ("train", "validation", "test")]
    assert len(set.union(*splits)) == 1797
    assert all(splits[i].isdisjoint(splits[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(data[name], repeated[name])
        assert set(data["labels"][data[name]].tolist()) == set(range(10))
    assert data["data_sha256"] == repeated["data_sha256"]


@pytest.mark.parametrize("factory, expected_parameters", [
    (build_mlp, 105214),
    (lambda: build_mlp("relu", batch_norm=True), 105622),
    (build_cnn, 61750),
])
def test_network_shapes_logits_and_gradient_flow(factory, expected_parameters):
    model = factory()
    images = torch.rand(8, 1, 28, 28)
    logits = model(images)
    assert logits.shape == (8, 10)
    assert sum(p.numel() for p in model.parameters()) == expected_parameters
    assert not any(isinstance(layer, (nn.Softmax, nn.LogSoftmax)) for layer in model.modules())
    nn.functional.cross_entropy(logits, torch.arange(8)).backward()
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            assert torch.isfinite(layer.weight.grad).all()
            assert layer.weight.grad.abs().sum() > 0


def test_mlp_comparisons_start_with_identical_dense_weights():
    weights = []
    for activation, batch_norm in [("sigmoid", False), ("relu", False), ("relu", True)]:
        torch.manual_seed(42)
        model = build_mlp(activation, batch_norm)
        weights.append([layer.weight.detach().clone() for layer in model if isinstance(layer, nn.Linear)])
    for other in weights[1:]:
        for first, second in zip(weights[0], other):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_evaluation_weights_partial_batches_by_rows_and_preserves_state():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 3))
    images = torch.randn(7, 2)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    model.eval()
    with torch.no_grad():
        logits = model(images)
        expected_loss = nn.functional.cross_entropy(logits, labels).item()
    model.train()
    before = deepcopy(model.state_dict())
    result = evaluate(model, images, labels, batch_size=3)
    assert result["loss"] == pytest.approx(expected_loss)
    np.testing.assert_array_equal(result["predictions"], logits.argmax(1).numpy())
    assert result["accuracy"] == pytest.approx(np.mean(result["predictions"] == labels.numpy()))
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_training_learns_a_known_boundary():
    images = torch.tensor([[-2., -1.], [-1., -2.], [-1., -1.], [-2., -2.],
                           [1., 1.], [1., 2.], [2., 1.], [2., 2.]])
    labels = torch.tensor([0] * 4 + [1] * 4)
    validation = torch.tensor([[-1.5, -1.5], [1.5, 1.5]])
    validation_labels = torch.tensor([0, 1])
    fitted = train_model(nn.Linear(2, 2), images, labels, validation, validation_labels,
                         epochs=8, learning_rate=0.1, batch_size=4)
    assert fitted["history"][-1]["train_loss"] < fitted["history"][0]["train_loss"] * 0.2
    assert evaluate(fitted["model"], validation, validation_labels)["accuracy"] == 1
    assert not fitted["model"].training


def test_best_checkpoint_restores_weights_and_batchnorm_buffers(monkeypatch):
    module = import_module("neural_network_classification.training")
    images, labels = torch.randn(8, 2), torch.tensor([0, 1] * 4)
    validation, validation_labels = torch.randn(4, 2), torch.tensor([0, 1] * 2)
    model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2))
    states, validation_losses = [], iter([3.0, 0.5, 1.0, 0.8])
    real_evaluate = module.evaluate

    def controlled_validation(model, X, y):
        result = real_evaluate(model, X, y)
        if X is validation:
            states.append(deepcopy(model.state_dict()))
            result["loss"] = next(validation_losses)
        return result

    monkeypatch.setattr(module, "evaluate", controlled_validation)
    fitted = module.train_model(model, images, labels, validation, validation_labels,
                                epochs=3, learning_rate=0.05, batch_size=4)
    assert fitted["best_epoch"] == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, states[1][name], rtol=0, atol=0)
    assert not torch.equal(states[1]["0.weight"], states[-1]["0.weight"])


def test_training_is_repeatable_with_fixed_initialization_and_shuffle_seed():
    images, labels = torch.randn(12, 2), torch.tensor([0, 1] * 6)
    histories, states = [], []
    for _ in range(2):
        torch.manual_seed(11)
        model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
        fitted = train_model(model, images[:8], labels[:8], images[8:], labels[8:],
                             epochs=3, batch_size=4, seed=20)
        histories.append(fitted["history"])
        states.append(deepcopy(model.state_dict()))
    assert histories[0] == histories[1]
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


def test_test_labels_do_not_influence_training_or_model_selection(monkeypatch):
    module = import_module("neural_network_classification.run_experiment")
    images, labels = torch.rand(40, 1, 28, 28), torch.arange(40) % 10
    data = {"images": images, "labels": labels, "train": np.arange(20),
            "validation": np.arange(20, 30), "test": np.arange(30, 40), "data_sha256": "test-only"}
    monkeypatch.setattr(module, "load_data", lambda seed: data)
    first = module.run_experiment(epochs=2)
    data["labels"] = labels.clone()
    data["labels"][30:] = (labels[30:] + 1) % 10
    second = module.run_experiment(epochs=2)
    assert first["history"].equals(second["history"])
    assert first["metadata"]["selected_model"] == second["metadata"]["selected_model"]
    assert first["predictions"].drop(columns="actual").equals(second["predictions"].drop(columns="actual"))
    expected = first["metrics"].sort_values("validation_loss", kind="stable").iloc[0].model
    assert first["metadata"]["selected_model"] == expected


def test_invalid_training_settings_and_singleton_batchnorm_batch_are_rejected():
    images, labels = torch.randn(5, 2), torch.tensor([0, 1, 0, 1, 0])
    for settings in [{"epochs": 0}, {"epochs": True}, {"learning_rate": 0},
                     {"learning_rate": float("nan")}, {"batch_size": 1}]:
        with pytest.raises(ValueError):
            train_model(nn.Linear(2, 2), images, labels, images, labels, **settings)
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    with pytest.raises(ValueError, match="BatchNorm"):
        train_model(model, images, labels, images, labels, batch_size=4)
