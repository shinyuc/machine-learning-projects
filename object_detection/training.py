"""CPU training with validation-only checkpoint selection."""

from copy import deepcopy
import math
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from .data import VehicleDataset, collate
from .evaluation import evaluate_detections
from .yolo import detection_loss, detect


@torch.no_grad()
def evaluate(model, data, indices, batch_size=8):
    loader = DataLoader(VehicleDataset(data, indices), batch_size=batch_size, collate_fn=collate)
    predictions, targets = [], []
    for images, boxes in loader:
        predictions.extend(detect(model, images))
        targets.extend(boxes)
    return evaluate_detections(predictions, targets)


def train_model(model, data, config, epochs=30, batch_size=8, seed=42):
    dataset = VehicleDataset(data, data["train"], training=True,
                             enhanced=config["enhanced"], seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=generator, collate_fn=collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=0.9, weight_decay=0.0005)
    history, best_ap, best_state, best_epoch = [], -1.0, None, 0
    # A short warmup for the compact run; no schedule tuning on test results.
    warmup_steps = min(1000, 5 * len(loader))
    freeze_epoch = epochs // 2 + 1 if config["freeze"] else None
    step = 0
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        if epoch == freeze_epoch:
            model.freeze_stem()
        dataset.epoch = epoch
        model.train()
        sums = {key: 0.0 for key in ("loss", "box_loss", "object_loss", "background_loss", "class_loss")}
        for images, boxes in loader:
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * min(1.0, step / warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = detection_loss(model(images), boxes, model.anchors)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; no results were exported.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, error_if_nonfinite=True)
            optimizer.step()
            for key, value in {"loss": float(loss.detach()), **parts}.items():
                sums[key] += value * len(images)
        validation = evaluate(model, data, data["validation"], batch_size)
        row = {"epoch": epoch, **{key: value / len(dataset) for key, value in sums.items()},
               "validation_ap50": validation["ap50"], "stem_frozen": epoch >= freeze_epoch if freeze_epoch else False}
        if not all(math.isfinite(row[key]) for key in sums):
            raise RuntimeError("Non-finite epoch metrics.")
        history.append(row)
        if validation["ap50"] > best_ap:
            best_ap, best_epoch, best_state = validation["ap50"], epoch, deepcopy(model.state_dict())
        print(f"{config['name']}: epoch {epoch:02d}/{epochs}, loss={row['loss']:.3f}, "
              f"validation AP50={validation['ap50']:.3f}", flush=True)
    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "validation_ap50": best_ap, "seconds": perf_counter() - started,
            "warmup_steps": warmup_steps, "freeze_epoch": freeze_epoch}
