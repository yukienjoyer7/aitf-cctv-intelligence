"""Basic profile of one CCTV-SR dataset as shipped by Roboflow: sizes, class balance and box sizes.

Reports images and boxes per provided split, polygon labels, empty images, per-class box counts, boxes per image
and box size in pixels (sqrt of box area, on the 512x512 image) overall and per class.

Usage: python benchmarks/cctv_sr/profile_dataset.py data/indoor      (or data/outdoor)
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from common import IMG_SIZE, SPLITS, class_names, read_label_file


def main(dataset_dir: Path) -> None:
    names = class_names(dataset_dir)
    boxes_per_image = {}  # (split, image name) -> number of boxes
    records = []  # (split, class_id, w, h) with w, h normalised
    n_polygons = 0
    for split in SPLITS:
        for image in sorted((dataset_dir / split / "images").glob("*.*")):
            label_file = dataset_dir / split / "labels" / (image.stem + ".txt")
            lines = label_file.read_text().splitlines() if label_file.exists() else []
            n_polygons += sum(len(line.split()) > 5 for line in lines)  # more than "cls cx cy w h" = polygon
            labels = read_label_file(label_file) if lines else []
            boxes_per_image[(split, image.name)] = len(labels)
            records += [(split, class_id, box[2], box[3]) for class_id, box in labels]

    n_images = Counter(split for split, _ in boxes_per_image)
    print("images per split:", dict(n_images), "| boxes:", len(records), "| polygon labels:", n_polygons)
    print("empty images:", sum(v == 0 for v in boxes_per_image.values()))

    per_split = Counter((split, class_id) for split, class_id, _, _ in records)
    print(f"\n{'class':20s} {'train':>6s} {'valid':>6s} {'test':>6s} {'total':>6s}")
    for class_id, name in enumerate(names):
        counts = [per_split[(split, class_id)] for split in SPLITS]
        print(f"{name:20s} {counts[0]:6d} {counts[1]:6d} {counts[2]:6d} {sum(counts):6d}")

    per_image = np.array(list(boxes_per_image.values()))
    print(
        f"\nboxes/image: mean {per_image.mean():.1f}, median {np.median(per_image):.0f}, "
        f"min {per_image.min()}, max {per_image.max()}"
    )

    # Box size = sqrt(width * height) in pixels. Small boxes (< 16 px) are hard for any detector at 512 px.
    w = np.array([r[2] for r in records]) * IMG_SIZE
    h = np.array([r[3] for r in records]) * IMG_SIZE
    size = np.sqrt(w * h)
    classes = np.array([r[1] for r in records])
    print(
        f"box size px: median {np.median(size):.0f}, p5 {np.percentile(size, 5):.0f}, p95 {np.percentile(size, 95):.0f}; "
        f"< 16 px: {(size < 16).mean() * 100:.1f}%; < 32 px: {(size < 32).mean() * 100:.1f}%"
    )
    print(
        "per-class median box size px:",
        {names[i]: int(np.median(size[classes == i])) for i in range(len(names)) if (classes == i).any()},
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
