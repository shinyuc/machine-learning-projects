"""Readable network definitions: each layer appears in its forward-pass order.

See THIRD_PARTY_NOTICES.md for the earlier MathWorks tutorial background and
the LeCun et al. (1998) architectural reference for the LeNet-style CNN.
"""

from torch import nn


def build_mlp(activation="sigmoid", batch_norm=False):
    """Flatten -> 120 hidden units -> 84 hidden units -> ten class logits."""
    if activation not in {"sigmoid", "relu"}:
        raise ValueError("activation must be 'sigmoid' or 'relu'.")
    layers = [nn.Flatten()]
    for inputs, outputs in [(28 * 28, 120), (120, 84)]:
        layers.append(nn.Linear(inputs, outputs))
        if batch_norm:
            layers.append(nn.BatchNorm1d(outputs))
        layers.append(nn.Sigmoid() if activation == "sigmoid" else nn.ReLU())
    layers.append(nn.Linear(84, 10))
    # CrossEntropyLoss consumes logits, so no Softmax layer is needed here.
    return nn.Sequential(*layers)


def build_cnn():
    """A LeNet-style CNN for 28x28 inputs, using ReLU, max pooling, and BN.

    Spatial sizes: 28 -> 28 -> 14 -> 10 -> 5. This is a modernized small
    architecture, not an exact implementation of the historical LeNet-5.
    """
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )
