"""Audit a labeled Hugging Face imagefolder Parquet export and optional existing images."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABELS = {0: "not_smoking", 1: "smoking"}


def audit(parquet_root: Path, comparison_root: Path | None = None) -> dict:
    """Hash image bytes without altering data or claiming near-duplicate independence."""
    records = defaultdict(list)
    counts = Counter()
    for split in ("train", "test"):
        files = sorted((parquet_root / split).glob("*.parquet"))
        if not files:
            raise ValueError(f"No completed Parquet files in {parquet_root / split}")
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=16, columns=["image", "label"]):
                for row in batch.to_pylist():
                    image, label = row["image"], row["label"]
                    if label not in LABELS or not image or not image.get("bytes"):
                        raise ValueError(f"Unexpected or missing image/label in {file}")
                    counts[(split, LABELS[label])] += 1
                    digest = hashlib.sha256(image["bytes"]).hexdigest()
                    records[digest].append((split, LABELS[label], image.get("path")))
    if comparison_root is not None and not comparison_root.is_dir():
        raise ValueError("Comparison image directory does not exist")
    comparison_hashes = set()
    if comparison_root is not None:
        for path in comparison_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                comparison_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    duplicates = [items for items in records.values() if len(items) > 1]
    cross_split = [items for items in duplicates if len({item[0] for item in items}) > 1]
    conflicts = [items for items in duplicates if len({item[1] for item in items}) > 1]
    return {
        "counts": {split: {name: counts[(split, name)] for name in LABELS.values()} for split in ("train", "test")},
        "unique_image_hashes": len(records),
        "duplicate_hash_groups": len(duplicates),
        "cross_train_test_duplicate_hash_groups": len(cross_split),
        "conflicting_label_hash_groups": len(conflicts),
        "overlap_with_comparison_rows": {
            split: sum(
                1 for digest, items in records.items() if digest in comparison_hashes for item in items if item[0] == split
            )
            for split in ("train", "test")
        }
        if comparison_root is not None
        else None,
        "cross_train_test_duplicates": [items for items in cross_split],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_root", type=Path, help="Directory with train/ and test/ Parquet files")
    parser.add_argument("--compare-images", type=Path, help="Compare exact image hashes with another local dataset")
    args = parser.parse_args()
    print(json.dumps(audit(args.parquet_root, args.compare_images), indent=2))


if __name__ == "__main__":
    main()
