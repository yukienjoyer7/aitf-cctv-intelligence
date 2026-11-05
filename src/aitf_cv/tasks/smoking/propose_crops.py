"""Propose person crops for manual smoking-label review; does not train a model."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[4]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FIELDS = ["image", "crop", "detector_confidence", "xyxy", "reviewed_label", "review_status", "notes"]


def clipped_box(box, width, height):
    """Round outward and clamp a detector box, returning None for empty crops."""
    x1, y1, x2, y2 = box
    result = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(width, math.ceil(x2)), min(height, math.ceil(y2)))
    return result if result[2] > result[0] and result[3] > result[1] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Directory containing source images")
    parser.add_argument("weights", type=Path, help="Local Ultralytics person detector checkpoint")
    parser.add_argument("output", type=Path, help="New output directory under data/smoking/")
    parser.add_argument("--confidence", type=float, default=0.1, help="Candidate threshold, not calibrated (default 0.1)")
    parser.add_argument("--device", default="cpu", help="Ultralytics device; default cpu")
    args = parser.parse_args()
    if not args.images.is_dir() or not args.weights.is_file():
        parser.error("images directory and local weights file must exist")
    if not 0 < args.confidence <= 1:
        parser.error("confidence must be greater than zero and at most one")
    target = args.output.resolve()
    permitted = (ROOT / "data" / "smoking").resolve()
    if not target.is_relative_to(permitted) or target == permitted:
        parser.error("output must be a new directory inside data/smoking/")
    if target.exists():
        parser.error("output already exists; choose a new directory to avoid overwriting review work")
    images = sorted(p for p in args.images.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        parser.error("no supported images found")

    from ultralytics import YOLO

    detector = YOLO(str(args.weights))
    person_classes = [index for index, name in detector.names.items() if name.lower() == "person"]
    if len(person_classes) != 1:
        parser.error("checkpoint must expose exactly one person class")
    target.mkdir(parents=True)
    (target / "crops").mkdir()
    metadata = {
        "detector_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "detector_classes": person_classes,
        "confidence": args.confidence,
        "confidence_status": "uncalibrated",
        "source_images_root": str(args.images.resolve()),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    count = 0
    with (target / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for image_path in images:
            relative = image_path.relative_to(args.images).as_posix()
            try:
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                result = detector.predict(image, classes=person_classes, conf=args.confidence, device=args.device, verbose=False)[
                    0
                ]
                proposals = 0
                for box in result.boxes:
                    coords = clipped_box(box.xyxy[0].tolist(), image.width, image.height)
                    if coords is None:
                        continue
                    key = hashlib.sha256(relative.encode()).hexdigest()[:20]
                    crop = f"crops/{key}_{proposals:03d}.jpg"
                    image.crop(coords).save(target / crop, quality=95)
                    writer.writerow(
                        {
                            "image": relative,
                            "crop": crop,
                            "detector_confidence": float(box.conf[0]),
                            "xyxy": coords,
                            "review_status": "pending",
                        }
                    )
                    proposals += 1
                    count += 1
                if not proposals:
                    writer.writerow({"image": relative, "review_status": "no_person_detected"})
            except (OSError, ValueError) as exc:
                writer.writerow({"image": relative, "review_status": "read_error", "notes": str(exc)})
    print(f"Proposed {count} crops from {len(images)} images; review.csv requires human labels.")


if __name__ == "__main__":
    main()
