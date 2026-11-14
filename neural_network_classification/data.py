"""Load the locally installed digits example and create three fixed splits."""

import hashlib

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import interpolate


def load_data(seed=42):
    """Return CPU tensors and disjoint 60/20/20 train/validation/test row IDs.

    The 1,797 UCI digit images ship with scikit-learn. They are 8x8, not MNIST.
    Bilinear resizing to 28x28 matches the model input size but adds no detail.
    Dividing by the documented maximum of 16 fits no parameters on held-out data.
    """
    digits = load_digits()
    images = torch.tensor(digits.images[:, None], dtype=torch.float32) / 16
    images = interpolate(images, size=(28, 28), mode="bilinear", align_corners=False)
    labels = torch.tensor(digits.target, dtype=torch.long)
    development, test = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=digits.target, random_state=seed,
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=digits.target[development], random_state=seed,
    )
    fingerprint = hashlib.sha256(
        digits.data.astype("<i2").tobytes() + digits.target.astype("<i2").tobytes()
    ).hexdigest()
    return {"images": images, "labels": labels, "train": train,
            "validation": validation, "test": test, "data_sha256": fingerprint}
