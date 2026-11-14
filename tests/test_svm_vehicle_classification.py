"""Known-answer HOG checks, training boundaries, and safe experiment exports."""

import json

import numpy as np
from PIL import Image
import pytest
from sklearn.svm import SVC

from svm_vehicle_classification.data import load_local_images, make_demo_images
from svm_vehicle_classification.features import hog_features
import svm_vehicle_classification.run_experiment as experiment
from svm_vehicle_classification.run_experiment import run_experiment, save_results


def test_hog_constant_image_and_known_gradient_directions():
    np.testing.assert_array_equal(hog_features(np.ones((64, 64))), np.zeros(1764))
    horizontal = np.tile(np.arange(16), (16, 1))
    # One 2x2-cell block: a horizontal ramp votes only into the zero-degree bin.
    h = hog_features(horizontal).reshape(4, 9)
    np.testing.assert_allclose(h[:, 0], 0.5)
    np.testing.assert_array_equal(h[:, 1:], 0)
    # Ninety degrees lies halfway between the 80- and 100-degree bins.
    v = hog_features(horizontal.T).reshape(4, 9)
    np.testing.assert_allclose(v[:, 4:6], 1 / np.sqrt(8))
    np.testing.assert_array_equal(v[:, [0, 1, 2, 3, 6, 7, 8]], 0)
    np.testing.assert_allclose(hog_features(-horizontal), hog_features(horizontal))
    np.testing.assert_allclose(hog_features(10 * horizontal + 3), hog_features(horizontal))


def test_hog_rejects_invalid_images():
    for image in [np.zeros((8, 8)), np.zeros((63, 64)), np.zeros((64, 64, 3)),
                  np.full((64, 64), np.nan)]:
        with pytest.raises(ValueError):
            hog_features(image)


def test_local_loader_deduplicates_and_rejects_conflicting_labels(tmp_path):
    # Test-only images; no external dataset is required by the test suite.
    for folder, start in [("non-vehicles", 0), ("vehicles", 10)]:
        target = tmp_path / folder
        target.mkdir()
        for i in range(5):
            crop = Image.fromarray(np.full((64, 64), start + i, dtype=np.uint8))
            crop.save(target / f"{i}.png")
            crop.save(target / f"{i}-duplicate.png")
    images, labels = load_local_images(tmp_path, per_class=5)
    assert len(np.unique(images[:, 0, 0])) == 10
    np.testing.assert_array_equal(labels, [-1] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="unique"):
        load_local_images(tmp_path, per_class=6)
    # Both copies of this positive image now duplicate a negative image.
    for suffix in ["", "-duplicate"]:
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(tmp_path / "vehicles" / f"0{suffix}.png")
    with pytest.raises(ValueError, match="conflicting"):
        load_local_images(tmp_path, per_class=5)


def test_demo_is_reproducible_and_experiment_has_disjoint_stratified_split():
    first, labels = make_demo_images(per_class=20)
    second, _ = make_demo_images(per_class=20)
    np.testing.assert_array_equal(first, second)
    result = run_experiment(per_class=20)
    assert result["features"].shape == (40, 1764)
    train, test = result["train"], result["test"]
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(40))
    np.testing.assert_array_equal(np.unique(labels[test], return_counts=True)[1], [4, 4])
    for row in result["metrics"].itertuples():
        truth = result["predictions"]["actual"].to_numpy()
        predicted = result["predictions"][row.method].to_numpy()
        assert row.test_accuracy == pytest.approx(
            np.mean(predicted == truth))
        tp = np.sum((predicted == 1) & (truth == 1))
        fp = np.sum((predicted == 1) & (truth == -1))
        fn = np.sum((predicted == -1) & (truth == 1))
        assert row.test_f1 == pytest.approx(2 * tp / (2 * tp + fp + fn))
    for kernel, model in result["models"].items():
        assert isinstance(model, SVC)
        np.testing.assert_array_equal(result["predictions"][f"SVC {kernel}"],
                                      model.predict(result["features"][test]))
    assert all(row["fit_status"] == 0 for row in result["metadata"]["fits"])


def test_both_kernels_fit_only_the_training_rows(monkeypatch):
    fitted = []

    class RecordingSVC(SVC):
        def fit(self, X, y, **kwargs):
            fitted.append((self.kernel, X.copy(), y.copy()))
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(experiment, "SVC", RecordingSVC)
    result = run_experiment(per_class=10)
    assert [kernel for kernel, _, _ in fitted] == ["linear", "rbf"]
    for _, X, y in fitted:
        np.testing.assert_array_equal(X, result["features"][result["train"]])
        np.testing.assert_array_equal(y, result["labels"][result["train"]])


@pytest.mark.parametrize("local", [False, True])
def test_export_limits_image_illustrations_to_generated_data(tmp_path, monkeypatch, local):
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    # Reuse generated fixtures through the local-data branch without external data.
    monkeypatch.setattr(experiment, "load_local_images",
                        lambda root, per_class, seed: make_demo_images(per_class, seed))
    result = run_experiment(data_dir="private-image-folder" if local else None, per_class=10)
    expected = {"metrics.csv", "predictions.csv", "metadata.json", "confusion_matrices.png"}
    if not local:
        expected |= {"synthetic_examples.png", "feature_example.png"}
    assert save_results(result, tmp_path) == len(expected)
    assert {path.name for path in tmp_path.iterdir()} == expected
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "private-image-folder" not in json.dumps(metadata)
    assert metadata["estimator"] == "sklearn.svm.SVC"
    assert not plt.get_fignums()
