"""Image transforms with synchronized boxes in zero-based xyxy coordinates."""

import torch
from torch.nn.functional import interpolate


def horizontal_flip(image, boxes):
    flipped = boxes.clone()
    flipped[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
    return image.flip(-1), flipped


def crop(image, boxes, window, minimum_visibility=0.5):
    """Clip boxes to (left, top, width, height), dropping mostly cropped objects."""
    left, top, width, height = window
    if (min(left, top) < 0 or min(width, height) <= 0
            or left + width > image.shape[-1] or top + height > image.shape[-2]):
        raise ValueError("Crop window must be inside the image.")
    shifted = boxes - boxes.new_tensor([left, top, left, top])
    shifted[:, [0, 2]] = shifted[:, [0, 2]].clamp(0, width)
    shifted[:, [1, 3]] = shifted[:, [1, 3]].clamp(0, height)
    old_area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
    new_size = shifted[:, 2:] - shifted[:, :2]
    keep = (new_size.min(dim=1).values >= 1) & (new_size.prod(dim=1) >= minimum_visibility * old_area)
    return image[:, top:top + height, left:left + width].clone(), shifted[keep]


def resize(image, boxes, size=227):
    height, width = image.shape[-2:]
    resized = interpolate(image[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    return resized, boxes * boxes.new_tensor([size / width, size / height] * 2)


def augment(image, boxes, rng, enhanced=False):
    """Baseline flip; enhanced brightness/noise/erasing plus box-aware cropping.

    Erasing simulates occlusion and keeps the object's full box. Cropping
    changes the visible image extent, so its boxes are clipped and filtered.
    """
    image, boxes = image.clone(), boxes.clone()
    if rng.random() < 0.5:
        image, boxes = horizontal_flip(image, boxes)
    if enhanced:
        image = (image * float(rng.uniform(0.7, 1.1))).clamp(0, 1)
        noise = torch.from_numpy(rng.random(image.shape[-2:]))
        image[:, noise < 0.01] = 0
        image[:, noise > 0.99] = 1
        height, width = image.shape[-2:]
        if rng.random() < 0.25:
            h, w = max(1, height // 10), max(1, width // 10)
            y, x = int(rng.integers(0, height - h + 1)), int(rng.integers(0, width - w + 1))
            image[:, y:y + h, x:x + w] = float(rng.uniform(0, 1))
        if rng.random() < 0.25:
            h, w = int(height * rng.uniform(0.8, 1)), int(width * rng.uniform(0.8, 1))
            y, x = int(rng.integers(0, height - h + 1)), int(rng.integers(0, width - w + 1))
            image, boxes = crop(image, boxes, (x, y, w, h))
    return image, boxes
