"""Find candidate near-duplicate images with 64-bit difference hashes; review matches manually."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = image.tobytes()
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def read_hashes(root: Path):
    result, errors = [], []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            try:
                result.append((path.relative_to(root).as_posix(), dhash(path)))
            except (OSError, ValueError) as exc:
                errors.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    return result, errors


def bands(value):
    return [(part, (value >> (part * 16)) & 0xFFFF) for part in range(4)]


def audit(root: Path, comparison: Path | None = None, distance: int = 3):
    if not 0 <= distance <= 3:
        raise ValueError("Distance must be between 0 and 3 for four-band candidate lookup")
    primary, errors = read_hashes(root)
    if not primary:
        raise ValueError("No readable primary images")
    other, comparison_errors = read_hashes(comparison) if comparison is not None else ([], [])
    if comparison is not None and not other:
        raise ValueError("No readable comparison images")
    buckets = defaultdict(list)
    for index, (_, value) in enumerate(primary):
        for key in bands(value):
            buckets[key].append(index)
    pairs = []
    for index, (name, value) in enumerate(primary):
        candidates = {candidate for key in bands(value) for candidate in buckets[key] if candidate < index}
        for candidate in candidates:
            other_name, other_value = primary[candidate]
            bits = (value ^ other_value).bit_count()
            if bits <= distance and name.split("/", 1)[0] != other_name.split("/", 1)[0]:
                pairs.append({"image": name, "match": other_name, "distance": bits})
    matches = []
    for name, value in other:
        candidates = {candidate for key in bands(value) for candidate in buckets[key]}
        for candidate in candidates:
            primary_name, primary_value = primary[candidate]
            bits = (value ^ primary_value).bit_count()
            if bits <= distance:
                matches.append({"image": primary_name, "comparison": name, "distance": bits})
    return {
        "method": "64-bit dHash candidate pairs; not verified duplicates",
        "max_distance": distance,
        "primary_images": len(primary),
        "comparison_images": len(other),
        "unreadable_primary": errors,
        "unreadable_comparison": comparison_errors,
        "cross_split_candidate_pairs": len(pairs),
        "cross_split_affected_images": len({item[key] for item in pairs for key in ("image", "match")}),
        "comparison_candidate_pairs": len(matches),
        "comparison_affected_primary_images": len({item["image"] for item in matches}),
        "comparison_affected_primary_by_split": dict(Counter(item.split("/", 1)[0] for item in {m["image"] for m in matches})),
        "cross_split_candidates": pairs,
        "comparison_candidates": matches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Images grouped by supplied split")
    parser.add_argument("--compare-images", type=Path)
    parser.add_argument("--distance", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_root, args.compare_images, args.distance), indent=2))


if __name__ == "__main__":
    main()
