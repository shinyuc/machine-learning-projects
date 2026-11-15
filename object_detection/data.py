"""Local-only vehicle data; fixed splits and training-only anchor estimation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset

from .augmentation import augment, resize
from .prepare_data import DEFAULT_DATA


def load_data(root=DEFAULT_DATA, seed=42):
    root = Path(root)
    if not (root / "annotations.json").is_file():
        raise FileNotFoundError("Local data is missing. Follow object_detection/README.md to prepare it.")
    records = json.loads((root / "annotations.json").read_text())
    images, boxes, hashes = [], [], []
    fingerprint = hashlib.sha256()
    for record in records:
        name = record["file"]
        if Path(name).name != name or "\\" in name:
            raise ValueError("Image names must be basenames within the local images directory.")
        with Image.open(root / "images" / name) as image:
            pixels = np.asarray(image.convert("RGB")).copy()
        target = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        height, width = pixels.shape[:2]
        if (not torch.isfinite(target).all() or (target[:, :2] < 0).any()
                or (target[:, 2:] <= target[:, :2]).any()
                or (target[:, 2:] > target.new_tensor([width, height])).any()):
            raise ValueError("Invalid local bounding boxes.")
        pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        hashes.append(pixel_hash)
        fingerprint.update(pixel_hash.encode() + target.numpy().astype("<f4").tobytes())
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        tensor, target = resize(tensor, target)
        images.append(tensor)
        boxes.append(target)
    if len(images) < 15:
        raise ValueError("Use at least 15 images for three splits.")
    indices = np.arange(len(images))
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                             .split(indices, groups=hashes))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                  .split(development, groups=np.asarray(hashes)[development]))
    return {"images": torch.stack(images), "boxes": boxes, "train": development[tr],
            "validation": development[va], "test": test, "groups": hashes,
            "data_sha256": fingerprint.hexdigest()}


class VehicleDataset(Dataset):
    def __init__(self, data, indices, training=False, enhanced=False, seed=42):
        self.data, self.indices = data, list(indices)
        self.training, self.enhanced, self.seed, self.epoch = training, enhanced, seed, 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        image, boxes = self.data["images"][row].clone(), self.data["boxes"][row].clone()
        if self.training:
            rng = np.random.default_rng([self.seed, self.epoch, int(row)])
            image, boxes = augment(image, boxes, rng, self.enhanced)
            image, boxes = resize(image, boxes)
        return image, boxes


def collate(batch):
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


def estimate_anchors(data, count=6, seed=42):
    """Median clustering with distance 1-IoU of widths/heights, on training boxes only."""
    if count != 6:
        raise ValueError("The two detection heads require six anchors.")
    sizes = torch.cat([data["boxes"][i][:, 2:] - data["boxes"][i][:, :2] for i in data["train"]]).numpy()
    if len(sizes) < count:
        raise ValueError("Too few training boxes to estimate anchors.")
    rng = np.random.default_rng(seed)
    centers = sizes[rng.choice(len(sizes), count, replace=False)].copy()
    for _ in range(100):
        intersection = np.minimum(sizes[:, None], centers[None]).prod(axis=2)
        overlap = intersection / (sizes.prod(axis=1)[:, None] + centers.prod(axis=1)[None] - intersection)
        assignments = overlap.argmax(axis=1)
        updated = np.array([np.median(sizes[assignments == k], axis=0)
                            if np.any(assignments == k) else centers[k] for k in range(count)])
        if np.allclose(updated, centers):
            break
        centers = updated
    centers = centers[np.argsort(centers.prod(axis=1))]
    return torch.tensor(np.stack([centers[3:], centers[:3]]), dtype=torch.float32)
