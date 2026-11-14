"""Known-answer checks for categorical splits, recursion, and prediction."""

from math import log2

import numpy as np
import pandas as pd
import pytest

from decision_trees.run_experiment import make_demo_data, run_experiment
from decision_trees.tree import build_tree, entropy, information_gain, predict, predict_one


def test_entropy_has_known_values():
    assert entropy([]) == 0
    assert entropy(["same"] * 4) == 0
    assert entropy([1, 1, 2, 2]) == pytest.approx(1)
    assert entropy([1, 2, 3]) == pytest.approx(log2(3))


def test_information_gain_uses_subset_weights():
    data = pd.DataFrame({"feature": ["a", "a", "a", "b"], "label": [0, 0, 1, 1]})
    conditional = 0.75 * (-(2 / 3) * log2(2 / 3) - (1 / 3) * log2(1 / 3))
    assert information_gain(data, "feature") == pytest.approx(1 - conditional)


def test_multiway_split_preserves_actual_nonconsecutive_category_values():
    data = pd.DataFrame({"feature": [10, 10, 30, 30, 80, 80], "label": ["a", "a", "b", "b", "c", "c"]})
    tree = build_tree(data)
    assert set(tree["children"]) == {10, 30, 80}
    assert predict(tree, pd.DataFrame({"feature": [80, 10, 30]})) == ["c", "a", "b"]


def test_zero_gain_at_root_can_still_learn_xor():
    data = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "label": [0, 1, 1, 0]})
    assert information_gain(data, "a") == 0
    assert information_gain(data, "b") == 0
    tree = build_tree(data)
    assert tree["feature"] == "a"  # Equal gains preserve feature order.
    assert predict(tree, data.drop(columns="label")) == data.label.tolist()
    assert predict(build_tree(data, max_depth=1), data) != data.label.tolist()


def test_pure_exhausted_and_depth_limited_nodes_use_majority_labels():
    pure = build_tree(pd.DataFrame({"feature": [1, 2], "label": ["yes", "yes"]}))
    assert pure["feature"] is None
    tied = build_tree(pd.DataFrame({"label": ["second", "first"]}))
    assert tied["prediction"] == "second"  # First encountered label wins ties.
    assert predict(tied, pd.DataFrame(index=[0, 1])) == ["second", "second"]
    data = pd.DataFrame({"feature": [0, 1, 1], "label": ["no", "yes", "yes"]})
    baseline = build_tree(data, max_depth=0)
    assert baseline["feature"] is None
    assert predict(baseline, data) == ["yes", "yes", "yes"]


def test_unseen_category_uses_the_current_nodes_majority():
    data = pd.DataFrame({
        "group": ["a", "a", "a", "a", "b", "b", "b"],
        "detail": ["x", "x", "x", "y", "x", "x", "x"],
        "label": ["yes", "yes", "yes", "no", "no", "no", "no"],
    })
    tree = build_tree(data)
    assert tree["feature"] == "group"
    assert predict_one(tree, {"group": "unseen", "detail": "x"}) == "no"
    assert predict_one(tree, {"group": "a", "detail": "unseen"}) == "yes"
    assert predict_one(tree, {"group": "a", "detail": "y"}) == "no"


def test_unobserved_pandas_categories_do_not_create_empty_children():
    data = pd.DataFrame({"feature": pd.Categorical(["a", "b"], categories=["a", "b", "unused"]), "label": [0, 1]})
    tree = build_tree(data)
    assert set(tree["children"]) == {"a", "b"}
    assert predict(tree, data) == [0, 1]


def test_target_is_not_a_predictor_and_training_data_is_unchanged():
    data = pd.DataFrame({"feature": [0, 0, 1, 1], "label": ["no", "no", "yes", "yes"]})
    before = data.copy(deep=True)
    tree = build_tree(data)
    pd.testing.assert_frame_equal(data, before)
    queries = data.copy()
    queries["label"] = "wrong"
    assert predict(tree, queries) == ["no", "no", "yes", "yes"]


def test_invalid_data_and_depth_raise_clear_errors():
    with pytest.raises(ValueError, match="target"):
        build_tree(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="rows"):
        build_tree(pd.DataFrame(columns=["x", "label"]))
    with pytest.raises(ValueError, match="missing"):
        build_tree(pd.DataFrame({"x": [None], "label": [1]}))
    with pytest.raises(ValueError, match="unique"):
        build_tree(pd.DataFrame([[1, 2, 3]], columns=["x", "x", "label"]))
    for value in [-1, 0.5, True]:
        with pytest.raises(ValueError, match="max_depth"):
            build_tree(pd.DataFrame({"x": [1], "label": [1]}), max_depth=value)


def test_demo_split_and_learned_rule():
    result = run_experiment()
    train, test = result["train"], result["test"]
    assert len(train) == 108 and len(test) == 36
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(range(144))
    data = make_demo_data()
    assert not data.drop(columns="label").duplicated().any()
    features = data.drop(columns="label")
    expected = ["walk" if row.weather == "cloudy" or (row.weather == "sunny" and row.wind == "calm") else "indoor"
                for row in features.itertuples(index=False)]
    assert predict(result["tree"], features) == expected
    for name in ["ID3", "One-split tree", "Majority baseline"]:
        accuracy = np.mean(result["predictions"][name] == result["predictions"]["actual"])
        reported = result["metrics"].set_index("method").loc[name, "test_accuracy"]
        assert accuracy == pytest.approx(reported)
