"""Single-class, all-points interpolated AP at IoU 0.5 (not COCO mAP)."""

import numpy as np
import torch
from torchvision.ops import box_iou


def evaluate_detections(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have one entry per image.")
    total_objects = sum(len(boxes) for boxes in targets)
    if total_objects == 0:
        raise ValueError("AP is undefined without ground-truth objects.")
    candidates = [(float(score), image, box) for image, pred in enumerate(predictions)
                  for box, score in zip(pred["boxes"], pred["scores"])]
    candidates.sort(key=lambda item: -item[0])
    matched = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in targets]
    true_positive = np.zeros(len(candidates))
    for index, (_, image, box) in enumerate(candidates):
        if not len(targets[image]):
            continue
        overlaps = box_iou(box[None], targets[image])[0]
        best = int(overlaps.argmax())
        if overlaps[best] >= iou_threshold and not matched[image][best]:
            true_positive[index] = 1
            matched[image][best] = True
    tp = np.cumsum(true_positive)
    fp = np.cumsum(1 - true_positive)
    recall = tp / total_objects
    precision = tp / np.maximum(tp + fp, 1)
    padded_recall = np.r_[0, recall, 1]
    envelope = np.r_[0, precision, 0]
    envelope = np.maximum.accumulate(envelope[::-1])[::-1]
    ap = float(np.sum(np.diff(padded_recall) * envelope[1:]))
    scores = np.array([item[0] for item in candidates])
    above = scores >= score_threshold
    threshold_tp = int(true_positive[above].sum())
    count = int(above.sum())
    return {"ap50": ap, "precision": threshold_tp / count if count else 0.0,
            "recall": threshold_tp / total_objects, "true_positives": threshold_tp,
            "false_positives": count - threshold_tp, "objects": total_objects,
            "curve": {"recall": recall, "precision": precision}}
