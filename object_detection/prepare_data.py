"""Convert the supplied vehicle annotations locally; never download or publish images."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.io import loadmat

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "vehicle-detection"


def read_vehicle_table(path):
    """Read the specific MATLAB example table, not arbitrary MATLAB objects.

    The MCOS table stores its columns in the MAT-file's function workspace.
    SciPy's low-level reader is needed for this older format. Names and shapes
    are checked before accepting it; see README for a plain-JSON alternative.
    """
    from scipy.io.matlab._mio5 import MatFile5Reader

    mat = loadmat(path)
    try:
        workspace = mat["__function_workspace__"].tobytes()
        if workspace[2:4] != b"IM":
            raise ValueError("Expected the little-endian example MAT-file.")
        stream = BytesIO(workspace)
        reader = MatFile5Reader(stream, byte_order="<", struct_as_record=False, squeeze_me=True)
        reader.initialize_read()
        stream.seek(8)
        header, _ = reader.read_var_header()
        wrapper = reader.read_var_array(header).MCOS
        payload = wrapper[wrapper.dtype.names[-1]][0]
        names, boxes = payload[2]
        if list(payload[7]) != ["imageFilename", "vehicle"] or len(names) != len(boxes):
            raise ValueError("Unexpected table columns.")
    except (KeyError, AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Unsupported annotation table; see the documented JSON format.") from exc
    return [(str(name), np.asarray(box, dtype=float).reshape(-1, 4)) for name, box in zip(names, boxes)]


def prepare_data(source, output=DEFAULT_DATA):
    """Copy authorized local images and convert 1-based xywh to 0-based xyxy."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source:
        raise ValueError("Choose a separate local output directory.")
    repository = Path(__file__).resolve().parents[1]
    ignored_roots = [repository / name for name in ("data", "datasets", "work")]
    if output.is_relative_to(repository) and not any(output.is_relative_to(root) for root in ignored_roots):
        raise ValueError("Inside this repository, store local images only under data/, datasets/, or work/.")
    rows = read_vehicle_table(source / "vehicleDatasetGroundTruth.mat")
    records, checked = [], []
    for name, xywh in rows:
        # Only the known image directory is read; stored records never use absolute paths.
        filename = Path(name.replace("\\", "/")).name
        path = source / "vehicleImages" / filename
        with Image.open(path) as image:
            width, height = image.size
        boxes = xywh.copy()
        boxes[:, :2] -= 1
        boxes[:, 2:] += boxes[:, :2]
        if (not np.isfinite(boxes).all() or (boxes[:, :2] < 0).any()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
                or (boxes[:, 2:] > [width, height]).any()):
            raise ValueError("Invalid annotation bounds; review the source table.")
        records.append({"file": filename, "boxes": boxes.tolist()})
        checked.append(path)
    if len({row["file"] for row in records}) != len(records):
        raise ValueError("Duplicate image names in the annotation table.")
    (output / "images").mkdir(parents=True, exist_ok=True)
    for path in checked:
        shutil.copy2(path, output / "images" / path.name)
    (output / "annotations.json").write_text(json.dumps(records, indent=2) + "\n")
    return len(records), sum(len(row["boxes"]) for row in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Local directory containing vehicleImages/ and vehicleDatasetGroundTruth.mat.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    images, boxes = prepare_data(args.source, args.output)
    print(f"Prepared {images} local images and {boxes} boxes. Keep this data out of Git.")


if __name__ == "__main__":
    main()
