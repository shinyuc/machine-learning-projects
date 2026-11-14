"""Numerical and data-boundary checks, independent of the downloaded dataset."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LinearRegression

from linear_regression.algorithms import LinearModel, gradient_descent, half_mse, least_squares
from linear_regression.data import MinMaxScaler, load_advertising, split_indices


def test_least_squares_matches_sklearn_and_known_solution():
    X = np.random.default_rng(5).normal(size=(80, 3))
    y = 4 + X @ np.array([2.0, -3.0, 0.5])
    ours = least_squares(X, y)
    reference = LinearRegression().fit(X, y)
    assert_allclose(ours.coef, [2, -3, 0.5], atol=1e-12)
    assert_allclose(ours.intercept, 4, atol=1e-12)
    assert_allclose(ours.predict(X), reference.predict(X), atol=1e-12)


def test_rank_deficient_least_squares_still_predicts_correctly():
    x = np.arange(12, dtype=float)
    X = np.column_stack((x, 2 * x, np.ones(len(x))))
    y = 3 + 4 * x
    assert_allclose(least_squares(X, y).predict(X), y, atol=1e-12)


@pytest.mark.parametrize("batch_size", [1, 13, 100])
def test_gradient_descent_converges_on_known_problem(batch_size):
    X = np.random.default_rng(9).normal(size=(41, 2))
    y = 1.5 + X @ np.array([2.0, -0.7])
    fitted = gradient_descent(X, y, learning_rate=0.05, epochs=1000, batch_size=batch_size)
    assert_allclose(fitted.model.predict(X), y, atol=1e-6)
    assert fitted.loss_history[-1] < 1e-12


def test_last_partial_batch_is_used_and_normalized_by_its_actual_size():
    # X=0 makes the slope irrelevant. Each batch has residual -intercept.
    # Starting b=0, three updates on constant targets give b=1-(1-0.1)^3.
    X = np.zeros((5, 1))
    result = gradient_descent(X, np.ones(5), epochs=1, batch_size=2, learning_rate=0.1)
    assert_allclose(result.model.intercept, 1 - 0.9**3)
    assert result.updates_per_epoch == 3


def test_seed_and_history_do_not_change_the_fitted_model():
    X = np.random.default_rng(6).normal(size=(23, 2))
    y = X[:, 0] - X[:, 1]
    a = gradient_descent(X, y, epochs=20, batch_size=7, seed=12)
    b = gradient_descent(X, y, epochs=20, batch_size=7, seed=12, record_history=False)
    assert_array_equal(a.model.coef, b.model.coef)
    assert a.model.intercept == b.model.intercept
    assert len(a.loss_history) == 21
    assert len(b.loss_history) == 0


def test_scaling_handles_constant_feature_and_corrects_the_intercept():
    X = np.array([[10, 5], [20, 5], [30, 5]], dtype=float)
    scaler = MinMaxScaler.fit(X)
    scaled_model = LinearModel(np.array([4.0, 2.0]), 7.0)
    original = scaler.to_original_scale(scaled_model)
    new_X = np.array([[15, 5], [50, 8]], dtype=float)
    assert_allclose(original.predict(new_X), scaled_model.predict(scaler.transform(new_X)))
    assert_allclose(scaler.transform(X)[:, 1], 0)
    # A held-out extreme stays outside the training range; no refit or clipping.
    assert scaler.transform(new_X)[1, 0] == 2.0
    assert_array_equal(scaler.minimum, [10, 5])


def test_split_is_disjoint_exhaustive_and_reproducible():
    train, test = split_indices(200, seed=42)
    assert len(train) == 160 and len(test) == 40
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(range(200))
    assert_array_equal(test, split_indices(200, seed=42)[1])


def test_loader_ignores_row_ids_and_accepts_source_column_case(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"Unnamed: 0": [1, 2], "TV": [1, 2], "radio": [2, 3], "newspaper": [3, 4], "sales": [4, 5]}).to_csv(path, index=False)
    frame = load_advertising(path)
    assert list(frame.columns) == ["TV", "Radio", "Newspaper", "Sales"]
    assert frame.shape == (2, 4)


def test_loader_rejects_missing_values(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("TV,Radio,Newspaper,Sales\n1,2,3,4\n2,,4,5\n")
    with pytest.raises(ValueError, match="missing"):
        load_advertising(path)


def test_target_shape_validation_prevents_accidental_broadcasting():
    with pytest.raises(ValueError, match="1-D"):
        least_squares(np.ones((5, 2)), np.ones((5, 1)))
    with pytest.raises(ValueError):
        half_mse(np.ones(5), np.ones((5, 1)))


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"epochs": -1}, {"learning_rate": 0}])
def test_invalid_optimizer_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        gradient_descent(np.ones((3, 1)), np.ones(3), **kwargs)
