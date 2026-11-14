"""A compact grayscale HOG descriptor implemented with NumPy.

Method: Dalal and Triggs (2005), Histograms of Oriented Gradients for Human
Detection. See THIRD_PARTY_NOTICES.md for method and implementation history.
"""

import numpy as np


def hog_features(image, cell_size=8, bins=9):
    """Unsigned gradients, orientation interpolation, and 2x2-cell L2-Hys blocks.

    Blocks advance by one cell. Pixels vote into their own spatial cell only;
    this simple implementation does not interpolate between neighboring cells.
    A 64x64 image with defaults produces 7 * 7 * 4 * 9 = 1764 features.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Use a finite 2-D grayscale image.")
    if (not isinstance(cell_size, int) or cell_size < 1
            or not isinstance(bins, int) or bins < 2):
        raise ValueError("cell_size must be positive and bins must be at least two.")
    height, width = image.shape
    if min(height, width) < 2 * cell_size or height % cell_size or width % cell_size:
        raise ValueError("Image dimensions must be multiples of cell_size with at least 2x2 cells.")

    # 1. Compute the magnitude and unsigned direction of each pixel gradient.
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    position = (np.degrees(np.arctan2(gy, gx)) % 180) * bins / 180
    lower = np.floor(position).astype(int) % bins
    fraction = position - np.floor(position)

    # 2. Accumulate magnitude-weighted votes into adjacent orientation bins.
    rows, columns = height // cell_size, width // cell_size
    histograms = np.zeros((rows, columns, bins))
    for row in range(rows):
        for column in range(columns):
            area = np.s_[row * cell_size:(row + 1) * cell_size,
                         column * cell_size:(column + 1) * cell_size]
            low, weight, mag = lower[area].ravel(), fraction[area].ravel(), magnitude[area].ravel()
            histograms[row, column] = (
                np.bincount(low, weights=mag * (1 - weight), minlength=bins)
                + np.bincount((low + 1) % bins, weights=mag * weight, minlength=bins)
            )

    # 3. Normalize overlapping blocks, clip large entries, and normalize again.
    blocks = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            block = histograms[row:row + 2, column:column + 2].ravel()
            block = block / np.sqrt(block @ block + 1e-10)
            block = np.minimum(block, 0.2)
            blocks.append(block / np.sqrt(block @ block + 1e-10))
    return np.concatenate(blocks)


def extract_features(images):
    """Return one HOG feature vector per image."""
    return np.vstack([hog_features(image) for image in images])
