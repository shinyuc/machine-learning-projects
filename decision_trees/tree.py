"""ID3 with categorical features, ordinary dictionaries, and recursion.

Each node stores a majority prediction, a sample count, a splitting feature,
and child nodes keyed by actual category values. A leaf has feature=None.
"""

from collections import Counter
from math import log2


def entropy(labels):
    """Return label entropy in bits; an empty or pure set has entropy zero."""
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((count / total) * log2(count / total) for count in counts.values())


def information_gain(data, feature, target="label"):
    """Entropy before splitting minus the weighted entropy of each category."""
    if data.empty:
        return 0.0
    conditional_entropy = 0.0
    for _, subset in data.groupby(feature, sort=False, observed=True):
        conditional_entropy += len(subset) / len(data) * entropy(subset[target])
    return entropy(data[target]) - conditional_entropy


def build_tree(data, target="label", max_depth=None):
    """Fit ID3 to a DataFrame containing categorical features and a target.

    The root is at depth 0. max_depth=0 gives a majority-class leaf;
    max_depth=1 permits one split. None permits all remaining features.
    Gain ties use column order; majority ties use first label encountered.
    Missing values are rejected. Numeric values are treated as categories.
    """
    if data.empty or target not in data.columns:
        raise ValueError("Training data must contain rows and the target column.")
    if not data.columns.is_unique or data.isna().any().any():
        raise ValueError("Use unique column names and data without missing values.")
    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be a nonnegative integer or None.")
    features = [column for column in data.columns if column != target]

    def grow(subset, remaining_features, depth):
        # Every node keeps a fallback for categories not seen during training.
        majority = Counter(subset[target]).most_common(1)[0][0]
        node = {"prediction": majority, "samples": len(subset),
                "feature": None, "children": {}}
        if (subset[target].nunique() == 1 or not remaining_features
                or (max_depth is not None and depth >= max_depth)):
            return node

        best = max(remaining_features, key=lambda name: information_gain(subset, name, target))
        node["feature"] = best
        next_features = [name for name in remaining_features if name != best]
        # A zero-gain split can still expose a useful interaction, such as XOR.
        for value, group in subset.groupby(best, sort=False, observed=True):
            node["children"][value] = grow(group, next_features, depth + 1)
        return node

    return grow(data, features, 0)


def predict_one(tree, sample):
    """Follow category branches; an unseen value uses this node's majority."""
    node = tree
    while node["feature"] is not None:
        value = sample[node["feature"]]
        if value not in node["children"]:
            return node["prediction"]
        node = node["children"][value]
    return node["prediction"]


def predict(tree, data):
    """Predict one label per row; target columns, if present, are not read."""
    if data.isna().any().any():
        raise ValueError("Prediction data must not contain missing values.")
    return [predict_one(tree, row) for _, row in data.iterrows()]


def tree_to_text(tree):
    """Return a readable tree with actual category names on each branch."""
    lines = []

    def visit(node, prefix="", branch=""):
        if node["feature"] is None:
            label = f"predict {node['prediction']} (n={node['samples']})"
        else:
            label = (f"split {node['feature']} (n={node['samples']}, "
                     f"fallback={node['prediction']})")
        lines.append(prefix + branch + label)
        for value, child in node["children"].items():
            visit(child, prefix + "  ", f"[{value}] ")

    visit(tree)
    return "\n".join(lines)
