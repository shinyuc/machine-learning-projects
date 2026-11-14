"""Generated teaching images and optional, read-only loading of local crops."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def make_demo_images(per_class=100, seed=42):
    """Draw noisy vehicle-like silhouettes and other shapes; no external data."""
    if not isinstance(per_class, int) or per_class < 5:
        raise ValueError("Use at least five images per class.")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:64, :64]
    images, labels = [], []
    for label in (-1, 1):
        for _ in range(per_class):
            image = rng.uniform(0.08, 0.22) + np.zeros((64, 64))
            image += rng.normal(0, 0.025, image.shape)
            x, y = rng.integers(10, 18), rng.integers(29, 37)
            shade = rng.uniform(0.55, 0.95)
            if label == 1:
                width, height = rng.integers(31, 41), rng.integers(10, 15)
                image[y:y + height, x:x + width] = shade
                image[y - 10:y, x + 7:x + width - 7] = shade
                image[y - 8:y - 1, x + 10:x + width - 10] = 0.3
                for center in (x + 8, x + width - 8):
                    wheel = (xx - center)**2 + (yy - y - height)**2 <= 5**2
                    image[wheel] = 0.02
            elif rng.random() < 0.5:
                # A house-like shape with a triangular roof and a door.
                image[y - 8:y + 18, x + 5:x + 31] = shade
                roof = (yy >= y - 22) & (yy < y - 8) & (abs(xx - x - 18) <= yy - y + 22)
                image[roof] = shade
                image[y + 4:y + 18, x + 14:x + 22] = 0.08
            else:
                # A group of vertical posts, with varying height and spacing.
                for offset in (0, 12, 24):
                    top = rng.integers(10, 26)
                    image[top:54, x + offset:x + offset + 5] = shade
            images.append(np.clip(image, 0, 1))
            labels.append(label)
    return np.asarray(images), np.asarray(labels)


def load_local_images(root, per_class=100, seed=42):
    """Sample unique 64x64 crops from vehicles/ and non-vehicles/ recursively.

    Exact duplicates after grayscale conversion/resizing are skipped. Similar
    crops from the same video may remain; this is not a sequence-level split.
    Paths and image files are never exported by the experiment.
    """
    if not isinstance(per_class, int) or per_class < 5:
        raise ValueError("Use at least five images per class.")
    root, rng = Path(root), np.random.default_rng(seed)
    images, labels, seen = [], [], {}
    for folder, label in [("non-vehicles", -1), ("vehicles", 1)]:
        paths = sorted(p for p in (root / folder).rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and p.is_file())
        selected = 0
        for index in rng.permutation(len(paths)):
            with Image.open(paths[index]) as source:
                pixels = np.asarray(source.convert("L").resize((64, 64), Image.Resampling.BILINEAR))
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError("An identical image has conflicting class labels.")
                continue
            seen[digest] = label
            images.append(pixels.astype(float) / 255)
            labels.append(label)
            selected += 1
            if selected == per_class:
                break
        if selected < per_class:
            raise ValueError(f"Need {per_class} unique readable images in each class folder.")
    return np.asarray(images), np.asarray(labels)
