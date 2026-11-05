"""Audit exact image duplicates and supplied labels before making smoking dataset splits."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def audit(root: Path) -> dict:
    """Audit content hashes; this does not detect visually similar or related images."""
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError("No images found")
    by_hash = defaultdict(list)
    for path in images:
        by_hash[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.relative_to(root))
    duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
    cross_split = [paths for paths in duplicates if len({path.parts[0] for path in paths}) > 1]
    conflicts = [paths for paths in duplicates if len({p.parent.name.lower() for p in paths} & {"smoking", "notsmoking"}) > 1]
    return {
        "images": len(images),
        "unique_sha256": len(by_hash),
        "image_counts_by_folder": dict(sorted(Counter(str(p.relative_to(root).parent) for p in images).items())),
        "duplicate_hash_groups": len(duplicates),
        "cross_supplied_folder_duplicate_groups": len(cross_split),
        "cross_supplied_folder_breakdown": dict(
            sorted(Counter("/".join(sorted({p.parts[0] for p in paths})) for paths in cross_split).items())
        ),
        "conflicting_supplied_class_groups": len(conflicts),
        "duplicate_groups": [[str(p) for p in paths] for paths in duplicates],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_root), indent=2))


if __name__ == "__main__":
    main()
