# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
# Modifications copyright (c) 2026 shinyuc, also distributed under BSD-3-Clause.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""YOLOv3 decoding, anchor assignment, loss and single-class NMS.

Adapted from Western Digital's model.py and training.py, revision
5944f2eae2b8e1e64c7c50cd42aa5e97e9d0e98c. See THIRD_PARTY_NOTICES.md.
Changes: two dynamic grids, xyxy boxes, sigmoid-space center loss, decoded-box
ignore mask, empty targets, and torchvision NMS. This file is BSD-3-Clause.
"""

import torch
from torch.nn import functional as F
from torchvision.ops import box_iou, nms


def decode(raw, anchors, image_size=227):
    """Convert (B, A, H, W, 6) logits to pixel xyxy boxes and object*class scores."""
    height, width = raw.shape[2:4]
    y, x = torch.meshgrid(torch.arange(height, device=raw.device),
                          torch.arange(width, device=raw.device), indexing="ij")
    grid = torch.stack([x, y], dim=-1).to(raw.dtype)
    stride = raw.new_tensor([image_size / width, image_size / height])
    centers = (raw[..., :2].sigmoid() + grid) * stride
    sizes = raw[..., 2:4].clamp(-8, 8).exp() * anchors[None, :, None, None]
    boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=-1)
    scores = raw[..., 4].sigmoid() * raw[..., 5].sigmoid()
    return boxes, scores


def assign_targets(predictions, targets, anchors, image_size=227):
    """Assign each box to its best width/height anchor across both heads.

    A cell/anchor can represent one object. On a collision the first box is kept;
    this is an explicit limitation of this small, single-class implementation.
    """
    positive = [torch.zeros(p.shape[:-1], dtype=torch.bool, device=p.device) for p in predictions]
    encoded = [torch.zeros_like(p[..., :4]) for p in predictions]
    all_anchors = anchors.reshape(-1, 2)
    for batch, boxes in enumerate(targets):
        for box in boxes:
            center, size = (box[:2] + box[2:]) / 2, box[2:] - box[:2]
            intersection = torch.minimum(size, all_anchors).prod(dim=1)
            overlaps = intersection / (size.prod() + all_anchors.prod(dim=1) - intersection)
            best = int(overlaps.argmax())
            head, anchor = divmod(best, 3)
            height, width = predictions[head].shape[2:4]
            grid_center = center / image_size * center.new_tensor([width, height])
            x, y = grid_center.long().tolist()
            x, y = min(x, width - 1), min(y, height - 1)
            index = (batch, anchor, y, x)
            if positive[head][index]:
                continue
            positive[head][index] = True
            encoded[head][index] = torch.cat([grid_center - grid_center.floor(),
                                             (size / all_anchors[best]).log()])
    return positive, encoded


def detection_loss(predictions, targets, anchors, image_size=227):
    """5*coordinate + object + 0.2*background + class loss, averaged per image."""
    positive, encoded = assign_targets(predictions, targets, anchors, image_size)
    zero = predictions[0].sum() * 0
    coordinate, obj, background, classification = zero, zero, zero, zero
    for head, raw in enumerate(predictions):
        pos = positive[head]
        negative = ~pos
        # Ignore unmatched predictions already overlapping any real object.
        with torch.no_grad():
            decoded, _ = decode(raw.detach(), anchors[head], image_size)
            for batch, boxes in enumerate(targets):
                if len(boxes):
                    overlaps = box_iou(decoded[batch].reshape(-1, 4), boxes)
                    negative[batch] &= overlaps.max(dim=1).values.reshape(pos.shape[1:]) < 0.5
        if pos.any():
            values, truth = raw[pos], encoded[head][pos]
            coordinate = coordinate + F.mse_loss(values[:, :2].sigmoid(), truth[:, :2], reduction="sum")
            coordinate = coordinate + F.mse_loss(values[:, 2:4], truth[:, 2:4], reduction="sum")
            obj = obj + F.softplus(-values[:, 4]).sum()
            classification = classification + F.softplus(-values[:, 5]).sum()
        background = background + F.softplus(raw[..., 4][negative]).sum()
    batch_size = len(targets)
    total = (5 * coordinate + obj + 0.2 * background + classification) / batch_size
    parts = {"box_loss": float(coordinate.detach() / batch_size),
             "object_loss": float(obj.detach() / batch_size),
             "background_loss": float(background.detach() / batch_size),
             "class_loss": float(classification.detach() / batch_size)}
    return total, parts


@torch.no_grad()
def detect(model, images, score_threshold=0.001, nms_iou=0.5, max_detections=100):
    """Low score floor for AP; reporting precision/recall uses a separate cutoff."""
    model.eval()
    predictions = model(images)
    boxes, scores = zip(*(decode(p, a, model.image_size) for p, a in zip(predictions, model.anchors)))
    boxes = torch.cat([b.reshape(len(images), -1, 4) for b in boxes], dim=1)
    scores = torch.cat([s.reshape(len(images), -1) for s in scores], dim=1)
    detections = []
    for image_boxes, image_scores in zip(boxes, scores):
        image_boxes = image_boxes.clamp(0, model.image_size)
        keep = ((image_scores >= score_threshold) & torch.isfinite(image_scores)
                & torch.isfinite(image_boxes).all(dim=1)
                & (image_boxes[:, 2:] > image_boxes[:, :2]).all(dim=1))
        image_boxes, image_scores = image_boxes[keep], image_scores[keep]
        selected = nms(image_boxes, image_scores, nms_iou)[:max_detections]
        detections.append({"boxes": image_boxes[selected], "scores": image_scores[selected]})
    return detections
