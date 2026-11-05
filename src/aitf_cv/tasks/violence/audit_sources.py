"""Audit the Surveillance Camera Fight Dataset without creating a train/test split."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ENTRY = re.compile(r"^((?:no)?fi\d{3}):", re.IGNORECASE)


def audit(root: Path) -> dict:
    """Report source-index ambiguities and identical files without guessing provenance."""
    sources = defaultdict(list)
    current_source = None
    unparsed = []
    for line in (root / "videos.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(("https://", "http://")):
            current_source = line
        elif match := ENTRY.match(line):
            sources[match.group(1).lower()].append(current_source)
        elif line:
            unparsed.append(line)
    files = sorted((root / "fight").glob("*.mp4")) + sorted((root / "noFight").glob("*.mp4"))
    if not files:
        raise ValueError("No MP4 clips found in fight/ or noFight/")
    by_hash = defaultdict(list)
    by_id = defaultdict(list)
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash[digest].append(str(path.relative_to(root)))
        by_id[path.stem.lower()].append(str(path.relative_to(root)))
    unique_source_counts = Counter(
        sources[path.stem.lower()][0]
        for path in files
        if len(sources[path.stem.lower()]) == 1 and sources[path.stem.lower()][0] is not None
    )
    return {
        "clips": len(files),
        "class_counts": {
            "fight": len(list((root / "fight").glob("*.mp4"))),
            "noFight": len(list((root / "noFight").glob("*.mp4"))),
        },
        "uniquely_mapped_clips": sum(
            len(sources[path.stem.lower()]) == 1 and sources[path.stem.lower()][0] is not None for path in files
        ),
        "unmapped_clip_ids": sorted(path.stem for path in files if not sources[path.stem.lower()]),
        "ambiguous_clip_sources": {key: value for key, value in sorted(sources.items()) if len(value) > 1 or None in value},
        "distinct_indexed_sources": len(unique_source_counts),
        "largest_source_group": max(unique_source_counts.values(), default=0),
        "duplicate_clip_ids": {key: value for key, value in by_id.items() if len(value) > 1},
        "identical_file_groups": [value for value in by_hash.values() if len(value) > 1],
        "unparsed_index_lines": unparsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Local dataset clone containing videos.txt and clip folders")
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_root), indent=2))


if __name__ == "__main__":
    main()
