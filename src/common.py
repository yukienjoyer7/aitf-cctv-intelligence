"""Shared helpers for the CCTV-SR scripts: paths, constants, label parsing and box geometry."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DATASETS = ("indoor", "outdoor")
SPLITS = ("train", "valid", "test")  # the splits that ship with the Roboflow export
IMG_SIZE = 512  # all images are 512x512 (stretched) by the Roboflow export

# Models in plotting/reporting order, with their display names.
MODELS = ("yolo26n", "yolo26s", "rtdetr-l")
MODEL_LABELS = {"yolo26n": "YOLO26n", "yolo26s": "YOLO26s", "rtdetr-l": "RT-DETR-L"}

# Original class names that are NOT people (static scenery).
STATIC_PREFIXES = ("Window", "Door")


def class_names(dataset_dir: Path) -> list[str]:
    """Class names declared in a Roboflow `data.yaml`, indexed by class id."""
    return yaml.safe_load((Path(dataset_dir) / "data.yaml").read_text())["names"]


def is_person(class_name: str) -> bool:
    """True for the people classes (`Female_*` / `Male_*`), False for windows and doors."""
    return class_name.startswith(("Female_", "Male_"))


def parse_label_line(line: str) -> tuple[int, list[float]]:
    """Parse one YOLO label line into (class_id, [cx, cy, w, h]) with coordinates normalised to 0-1.

    The export mixes plain boxes (`cls cx cy w h`) with polygons (`cls x1 y1 x2 y2 ...`, 124 indoor and
    933 outdoor quads). Polygons are replaced by their enclosing axis-aligned box.
    """
    cls, *vals = line.split()
    vals = list(map(float, vals))
    if len(vals) > 4:
        xs, ys = vals[0::2], vals[1::2]
        vals = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
    return int(cls), vals


def read_label_file(path: Path) -> list[tuple[int, list[float]]]:
    """All (class_id, [cx, cy, w, h]) entries of a label file."""
    return [parse_label_line(line) for line in Path(path).read_text().splitlines()]


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    """Centre/size box -> corner box (x1, y1, x2, y2)."""
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two corner boxes."""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-12)
