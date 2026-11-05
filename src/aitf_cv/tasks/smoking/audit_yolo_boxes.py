"""Audit a local YOLO-format smoking dataset without training or modifying it."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXPECTED = {"cigarette", "non smoking person", "smoking person"}


def names_from_yaml(dataset: Path) -> dict[int, str]:
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(key): value for key, value in names.items()}
    else:
        raise ValueError("data.yaml must define a list or mapping of class names")
    if any(not isinstance(value, str) for value in names.values()):
        raise ValueError("Class names must be strings")
    if {value.strip().lower() for value in names.values()} != EXPECTED:
        raise ValueError("Dataset classes do not match the expected person/cigarette classes")
    return names


def audit(dataset: Path, comparison: Path | None = None) -> dict:
    """Report exact image duplication and malformed YOLO boxes, never infer missing labels."""
    names = names_from_yaml(dataset)
    if comparison is not None and not comparison.is_dir():
        raise ValueError("Comparison image directory does not exist")
    hashes = defaultdict(list)
    box_counts = Counter()
    image_counts = Counter()
    missing_labels, empty_labels, invalid_labels = [], [], []
    seen_splits = []
    for split in ("train", "valid", "val", "test"):
        image_dir = dataset / split / "images"
        if not image_dir.is_dir():
            continue
        if split == "val" and "valid" in seen_splits:
            raise ValueError("Both val and valid directories exist; resolve split ambiguity")
        seen_splits.append(split)
        for path in sorted(image_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(dataset).as_posix()
            image_counts[split] += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[digest].append((split, relative))
            label = dataset / split / "labels" / path.relative_to(image_dir).with_suffix(".txt")
            if not label.is_file():
                missing_labels.append(relative)
                continue
            lines = label.read_text(encoding="utf-8").splitlines()
            if not any(line.strip() for line in lines):
                empty_labels.append(relative)
                continue
            for index, line in enumerate(lines, 1):
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError("expected class and four coordinates")
                    class_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:])
                    if class_id not in names or not all(math.isfinite(v) for v in (x, y, width, height)):
                        raise ValueError("unknown class or non-finite coordinate")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("coordinates outside normalized range")
                    if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
                        raise ValueError("box extends beyond image")
                    box_counts[(split, names[class_id])] += 1
                except ValueError as exc:
                    invalid_labels.append({"file": label.relative_to(dataset).as_posix(), "line": index, "error": str(exc)})
    if "train" not in seen_splits or "test" not in seen_splits or not ({"val", "valid"} & set(seen_splits)):
        raise ValueError("Expected train, test and either valid or val image directories")
    if not sum(image_counts.values()):
        raise ValueError("No images found")
    duplicates = [items for items in hashes.values() if len(items) > 1]
    cross = [items for items in duplicates if len({item[0] for item in items}) > 1]
    comparison_hashes = set()
    if comparison:
        for path in comparison.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                comparison_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return {
        "classes": names,
        "image_counts": dict(image_counts),
        "box_counts": {split: {name: box_counts[(split, name)] for name in names.values()} for split in seen_splits},
        "missing_label_files": missing_labels,
        "empty_label_files": empty_labels,
        "invalid_labels": invalid_labels,
        "duplicate_hash_groups": len(duplicates),
        "cross_split_duplicate_hash_groups": len(cross),
        "cross_split_duplicates": cross,
        "overlap_comparison_images_by_split": (
            {
                split: sum(
                    1 for digest, entries in hashes.items() if digest in comparison_hashes for s, _ in entries if s == split
                )
                for split in seen_splits
            }
            if comparison is not None
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--compare-images", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_root, args.compare_images), indent=2))


if __name__ == "__main__":
    main()
