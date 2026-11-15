"""Detection correctness checks use generated fixtures, never the private photos."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")

from object_detection.augmentation import horizontal_flip, crop, resize, augment
from object_detection.data import load_data, estimate_anchors
from object_detection.evaluation import evaluate_detections
from object_detection.model import SqueezeYOLOv3
from object_detection.yolo import decode, assign_targets, detection_loss, detect
from object_detection import prepare_data


def anchors():
    return torch.tensor([[[60, 70], [80, 90], [100, 110]], [[10, 20], [20, 30], [30, 40]]], dtype=torch.float32)


def test_flip_crop_resize_keep_boxes_aligned():
    image = torch.arange(3 * 10 * 20).view(3, 10, 20).float()
    boxes = torch.tensor([[2., 1., 8., 7.], [15., 1., 19., 5.]])
    flipped, transformed = horizontal_flip(image, boxes)
    assert torch.equal(flipped, image.flip(-1))
    assert torch.equal(transformed, torch.tensor([[12., 1., 18., 7.], [1., 1., 5., 5.]]))
    clipped_image, clipped = crop(image, boxes, (4, 0, 8, 10))
    assert torch.equal(clipped, torch.tensor([[0., 1., 4., 7.]]))
    resized, scaled = resize(clipped_image, clipped, size=20)
    assert resized.shape == (3, 20, 20)
    assert torch.equal(scaled, torch.tensor([[0., 2., 10., 14.]]))
    assert torch.equal(boxes[0], torch.tensor([2., 1., 8., 7.]))


def test_augmentation_is_repeatable_and_does_not_modify_input():
    image = torch.ones(3, 30, 40)
    boxes = torch.tensor([[1., 1., 39., 29.]])
    first = augment(image, boxes, np.random.default_rng(6), enhanced=True)
    second = augment(image, boxes, np.random.default_rng(6), enhanced=True)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert image.eq(1).all()
    assert torch.equal(boxes, torch.tensor([[1., 1., 39., 29.]]))
    assert first[0].min() >= 0 and first[0].max() <= 1


def test_zero_logits_decode_to_cell_centers_and_anchor_sizes():
    raw = torch.zeros(1, 3, 2, 2, 6)
    boxes, scores = decode(raw, anchors()[0], image_size=100)
    expected_center = torch.tensor([25., 25.])
    assert torch.equal((boxes[0, 0, 0, 0, :2] + boxes[0, 0, 0, 0, 2:]) / 2, expected_center)
    assert torch.equal(boxes[0, 0, 0, 0, 2:] - boxes[0, 0, 0, 0, :2], torch.tensor([60., 70.]))
    assert scores.eq(0.25).all()


def test_anchor_assignment_selects_one_head_and_cell():
    raw = [torch.zeros(1, 3, n, n, 6) for n in (2, 4)]
    targets = [torch.tensor([[15., 10., 35., 40.]])]
    pos, encoded = assign_targets(raw, targets, anchors(), image_size=100)
    assert pos[0].sum() == 0 and pos[1].sum() == 1
    assert pos[1][0, 1, 1, 1]
    assert torch.equal(encoded[1][0, 1, 1, 1], torch.zeros(4))


def test_overlapping_unassigned_predictions_are_ignored_for_background_loss():
    same_anchors = torch.full((2, 3, 2), 10.)
    raw = [torch.zeros(1, 3, 1, 1, 6, requires_grad=True) for _ in range(2)]
    loss, _ = detection_loss(raw, [torch.tensor([[45., 45., 55., 55.]])], same_anchors, image_size=100)
    loss.backward()
    assert raw[0].grad[0, 0, 0, 0, 4] < 0  # assigned object
    assert raw[0].grad[0, 1:, 0, 0, 4].eq(0).all()  # same decoded box, unassigned
    assert raw[1].grad[..., 4].eq(0).all()


def test_empty_targets_have_finite_background_gradients():
    raw = [torch.zeros(1, 3, n, n, 6, requires_grad=True) for n in (2, 4)]
    loss, _ = detection_loss(raw, [torch.empty(0, 4)], anchors())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad[..., 4].gt(0).all() for p in raw)


def test_ap_known_answer_and_duplicate_detection():
    gt = [torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])]
    pred = [{"boxes": torch.tensor([[0., 0., 10., 10.], [40., 40., 50., 50.],
                                    [20., 20., 30., 30.], [0., 0., 10., 10.]]),
             "scores": torch.tensor([0.9, 0.8, 0.7, 0.6])}]
    result = evaluate_detections(pred, gt)
    assert result["ap50"] == pytest.approx(5 / 6)
    assert result["precision"] == 0.5 and result["recall"] == 1
    assert result["true_positives"] == 2 and result["false_positives"] == 2
    empty = evaluate_detections([{"boxes": torch.empty(0, 4), "scores": torch.empty(0)}], gt)
    assert empty["ap50"] == 0


def test_nms_suppresses_same_object_from_both_heads():
    class FixedModel:
        image_size = 100
        anchors = torch.full((2, 3, 2), 20.)

        def eval(self):
            return self

        def __call__(self, images):
            return [torch.zeros(1, 3, 1, 1, 6) for _ in range(2)]

    detections = detect(FixedModel(), torch.zeros(1, 3, 100, 100))
    assert len(detections[0]["boxes"]) == 1
    assert torch.equal(detections[0]["boxes"][0], torch.tensor([40., 40., 60., 60.]))


def test_model_shapes_finite_update_and_actual_stem_freeze():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = SqueezeYOLOv3(anchors())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005)
    model.freeze_stem()
    before = model.backbone[0].weight.detach().clone()
    head_before = model.outputs[0].bias.detach().clone()
    raw = model(torch.rand(1, 3, 227, 227))
    assert [tuple(p.shape) for p in raw] == [(1, 3, 14, 14, 6), (1, 3, 28, 28, 6)]
    loss, _ = detection_loss(raw, [torch.tensor([[40., 40., 100., 110.]])], model.anchors)
    loss.backward()
    assert model.backbone[0].weight.grad is None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert torch.equal(before, model.backbone[0].weight)
    assert not torch.equal(head_before, model.outputs[0].bias)


def test_duplicate_pixels_never_cross_splits(tmp_path):
    (tmp_path / "images").mkdir()
    records = []
    for i in range(40):
        filename = f"fixture_{i}.png"
        Image.new("RGB", (20, 10), color=(i % 20, 0, 0)).save(tmp_path / "images" / filename)
        records.append({"file": filename, "boxes": [[1, 1, 10, 8]]})
    (tmp_path / "annotations.json").write_text(json.dumps(records))
    data = load_data(tmp_path)
    groups = [{data["groups"][i] for i in data[key]} for key in ("train", "validation", "test")]
    assert len(data["images"]) == 40 and len(set(data["groups"])) == 20
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    repeated = load_data(tmp_path)
    assert np.array_equal(data["train"], repeated["train"])


def test_anchors_do_not_use_validation_or_test_boxes():
    boxes = [torch.tensor([[0., 0., 5. + i, 10. + i]]) for i in range(10)]
    data = {"boxes": boxes, "train": np.arange(8)}
    before = estimate_anchors(data)
    boxes[8] = torch.tensor([[0., 0., 999., 999.]])
    assert torch.equal(before, estimate_anchors(data))


def test_local_conversion_handles_one_based_coordinates(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "converted"
    (source / "vehicleImages").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(source / "vehicleImages" / "fixture.jpg")
    monkeypatch.setattr(prepare_data, "read_vehicle_table",
                        lambda _: [("vehicleImages/fixture.jpg", np.array([[1., 1., 20., 10.]]))])
    assert prepare_data.prepare_data(source, output) == (1, 1)
    records = json.loads((output / "annotations.json").read_text())
    assert records[0]["boxes"] == [[0, 0, 20, 10]]
    assert (output / "images" / "fixture.jpg").is_file()


def test_converter_rejects_a_public_repository_directory(tmp_path):
    public_directory = Path(prepare_data.__file__).resolve().parent / "photos"
    with pytest.raises(ValueError, match="store local images only"):
        prepare_data.prepare_data(tmp_path, public_directory)
    assert not public_directory.exists()
