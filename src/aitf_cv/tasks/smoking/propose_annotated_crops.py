"""Convert person-labeled YOLO boxes into crops for review, without training."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from audit_yolo_boxes import EXTENSIONS, audit, names_from_yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[4]
FIELDS = ["split", "image", "image_sha256", "crop", "source_label", "reviewed_label", "review_status", "xyxy"]


def bounds(x, y, width, height, image_width, image_height):
    return (
        max(0, math.floor((x - width / 2) * image_width)),
        max(0, math.floor((y - height / 2) * image_height)),
        min(image_width, math.ceil((x + width / 2) * image_width)),
        min(image_height, math.ceil((y + height / 2) * image_height)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Extracted YOLO dataset with data.yaml")
    parser.add_argument("output", type=Path, help="New review directory under data/smoking/")
    args = parser.parse_args()
    target = args.output.resolve()
    permitted = (ROOT / "data" / "smoking").resolve()
    if not target.is_relative_to(permitted) or target == permitted:
        parser.error("output must be a new directory inside data/smoking/")
    if target.exists():
        parser.error("output already exists; do not overwrite review work")
    report = audit(args.dataset_root)
    if report["invalid_labels"]:
        parser.error("invalid annotations found; run audit_yolo_boxes.py and resolve them first")
    names = names_from_yaml(args.dataset_root)
    relevant = {
        i: name.strip().lower() for i, name in names.items() if name.strip().lower() in {"smoking person", "non smoking person"}
    }
    target.mkdir(parents=True)
    (target / "crops").mkdir()
    (target / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    total = 0
    with (target / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for split in report["image_counts"]:
            image_dir = args.dataset_root / split / "images"
            for image_path in sorted(image_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in EXTENSIONS:
                    continue
                relative = image_path.relative_to(args.dataset_root).as_posix()
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                label_path = args.dataset_root / split / "labels" / image_path.relative_to(image_dir).with_suffix(".txt")
                if not label_path.is_file():
                    writer.writerow(
                        {"split": split, "image": relative, "image_sha256": digest, "review_status": "missing_annotation"}
                    )
                    continue
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                count = 0
                for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    class_id, *coords = line.split()
                    class_id = int(class_id)
                    if class_id not in relevant:
                        continue
                    box = bounds(*map(float, coords), image.width, image.height)
                    if box[0] >= box[2] or box[1] >= box[3]:
                        raise ValueError(f"Empty crop: {relative}, annotation {index + 1}")
                    key = hashlib.sha256(relative.encode()).hexdigest()[:20]
                    crop = f"crops/{key}_{index:04d}.jpg"
                    image.crop(box).save(target / crop, quality=95)
                    writer.writerow(
                        {
                            "split": split,
                            "image": relative,
                            "image_sha256": digest,
                            "crop": crop,
                            "source_label": relevant[class_id],
                            "review_status": "pending",
                            "xyxy": box,
                        }
                    )
                    count += 1
                    total += 1
                if count == 0:
                    writer.writerow(
                        {"split": split, "image": relative, "image_sha256": digest, "review_status": "no_person_annotation"}
                    )
    print(f"Created {total} person crops; source labels require review and splits require leakage checks.")


if __name__ == "__main__":
    main()
