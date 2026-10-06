"""Show why the provided CCTV-SR splits leak and how much of each dataset is static scenery.

Reports, for one dataset:
  * frame-id range and gaps (are the frames consecutive?), people per image, share of images with no person;
  * the share of boxes that are static windows/doors, and the Female/Male split;
  * STATIC-LABEL LEAKAGE: fraction of valid/test window/door boxes that have a same-class train box at almost
    the same pixel position (IoU >= 0.7), which a model can memorise;
  * FRAME LEAKAGE: how close each valid/test frame is to its nearest train frame, in pixels and in frame ids,
    compared with the pixel difference between two random train frames.

Usage: python src/profile_leakage.py data/indoor      (or data/outdoor)
"""

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import SPLITS, class_names, iou, is_person, read_label_file, xywh_to_xyxy

STATIC_IOU = 0.7  # "same position" threshold for static boxes
THUMB = 128  # frames are compared as 128x128 grayscale thumbnails
CLOSE_IDS = 5  # a train frame within this many frame ids counts as "right next to" the test frame


def load(dataset_dir: Path, names: list[str]):
    """frames[(split, image name)] = (image path, [(class_id, corner box)]); frame_id[...] = numeric frame id."""
    frames, frame_id = {}, {}
    for split in SPLITS:
        for image in sorted((dataset_dir / split / "images").glob("*.jpg")):
            label_file = dataset_dir / split / "labels" / (image.stem + ".txt")
            boxes = [(c, xywh_to_xyxy(b)) for c, b in read_label_file(label_file)]
            frames[(split, image.name)] = (image, boxes)
            frame_id[(split, image.name)] = int(re.search(r"frame_0*(\d+)", image.name).group(1))
    return frames, frame_id


def thumbnail(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((THUMB, THUMB)), np.float32)


def main(dataset_dir: Path) -> None:
    names = class_names(dataset_dir)
    people = {i for i, n in enumerate(names) if is_person(n)}
    frames, frame_id = load(dataset_dir, names)

    # --- frame ids: are the frames consecutive? ---
    ids = sorted(frame_id.values())
    gaps = [(a, b) for a, b in zip(ids, ids[1:], strict=False) if b - a > 20]
    print("frame id range", ids[0], "-", ids[-1], "| distinct ids", len(set(ids)), "of", len(ids), "| id gaps > 20:", gaps)

    # --- people per image ---
    per_image = {k: sum(c in people for c, _ in boxes) for k, (_, boxes) in frames.items()}
    for split in SPLITS:
        x = np.array([v for k, v in per_image.items() if k[0] == split])
        print(f"{split}: people/img mean {x.mean():.1f}, images with 0 people {np.mean(x == 0) * 100:.0f}%, max {x.max()}")

    # --- static scenery share and gender split ---
    all_classes = [c for _, boxes in frames.values() for c, _ in boxes]
    n_static = sum(c not in people for c in all_classes)
    n_people = len(all_classes) - n_static
    n_female = sum(c in people and names[c].startswith("Female") for c in all_classes)
    n_male = n_people - n_female
    print(f"static (window/door) boxes: {n_static}/{len(all_classes)} = {n_static / len(all_classes) * 100:.0f}% of all boxes")
    print(
        f"people boxes: {n_people} (Female {n_female} = {n_female / n_people * 100:.0f}%, Male {n_male} = {n_male / n_people * 100:.0f}%)"
    )

    # --- static-label leakage: do valid/test window boxes sit where a train window box sits? ---
    train_static = [(c, b) for k, (_, boxes) in frames.items() if k[0] == "train" for c, b in boxes if c not in people]
    for split in ("valid", "test"):
        queries = [(c, b) for k, (_, boxes) in frames.items() if k[0] == split for c, b in boxes if c not in people]
        hit = np.mean([any(c == c2 and iou(b, b2) >= STATIC_IOU for c2, b2 in train_static) for c, b in queries])
        print(f"{split}: {hit * 100:.0f}% of static boxes have a same-class TRAIN box with IoU>={STATIC_IOU} (n={len(queries)})")

    # --- frame leakage: nearest train frame in pixel space ---
    thumbs = {k: thumbnail(path) for k, (path, _) in frames.items()}
    train_keys = [k for k in frames if k[0] == "train"]
    train_stack = np.stack([thumbs[k] for k in train_keys])
    for split in ("valid", "test"):
        diff, id_dist = [], []
        for k in (k for k in frames if k[0] == split):
            d = np.abs(train_stack - thumbs[k]).mean((1, 2))  # mean abs pixel difference (0-255) to every train frame
            nearest = d.argmin()
            diff.append(d[nearest])
            id_dist.append(abs(frame_id[train_keys[nearest]] - frame_id[k]))
        id_dist = np.array(id_dist)
        print(
            f"{split}: mean abs pixel diff to nearest train frame: median {np.median(diff):.1f}; "
            f"nearest-train frame-id distance: median {np.median(id_dist):.0f}, "
            f"within {CLOSE_IDS} ids: {np.mean(id_dist <= CLOSE_IDS) * 100:.0f}%"
        )
    # reference: how different are two RANDOM train frames? (the baseline for "near-duplicate")
    rng = np.random.default_rng(0)
    a, b = rng.choice(len(train_keys), 200), rng.choice(len(train_keys), 200)
    ref = [np.abs(train_stack[i] - train_stack[j]).mean() for i, j in zip(a, b, strict=True) if i != j]
    print("reference: random train-vs-train mean abs diff median", round(float(np.median(ref)), 1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
