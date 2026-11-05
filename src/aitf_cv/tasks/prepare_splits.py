"""Prepare provisional grouped split manifests for smoking or violence research.

Source annotations are not verified ground truth. This script does not train a model.
Run as PYTHONPATH=src python -m aitf_cv.tasks.prepare_splits <task> <output>.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from aitf_cv.tasks.smoking.audit_near_duplicates import bands, read_hashes
from aitf_cv.tasks.split_utils import FRACTIONS, SPLITS, Groups, choose_group_split
from aitf_cv.tasks.violence.audit_sources import ENTRY

ROOT = Path(__file__).resolve().parents[3]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_identifier(items):
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()[:20]


def smoking_records():
    """Group source images by filename and perceptual candidates, and retain source-label provenance."""
    dataset = ROOT / "data/smoking/smoking_people_v2"
    review = ROOT / "data/smoking/smoking_people_review_v1"
    source_csv = review / "review.csv"
    images, errors = read_hashes(dataset)
    if errors or not images:
        raise ValueError(f"Source images unavailable or unreadable: {len(errors)} errors")
    groups = Groups(name for name, _ in images)
    buckets = defaultdict(list)
    base_names = {}
    candidate_pairs = 0
    for name, value in images:
        base = Path(name).name.split(".rf.")[0]
        if base in base_names:
            groups.union(name, base_names[base])
        else:
            base_names[base] = name
        candidates = {item for key in bands(value) for item in buckets[key]}
        for other_name, other_hash in candidates:
            if (value ^ other_hash).bit_count() <= 3:
                groups.union(name, other_name)
                candidate_pairs += 1
        for key in bands(value):
            buckets[key].append((name, value))
    group_names = defaultdict(list)
    for name, _ in images:
        group_names[groups.find(name)].append(name)
    ids = {root: group_identifier(members) for root, members in group_names.items()}
    prepared, excluded = [], []
    omitted_review_rows = Counter()
    labels = {"smoking person": "smoking", "non smoking person": "non_smoking"}
    with source_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["crop"]:
                omitted_review_rows[row["review_status"]] += 1
                continue
            crop = review / row["crop"]
            image = dataset / row["image"]
            if row["image"] not in groups.parent or not image.is_file() or not crop.is_file():
                raise ValueError("Review manifest references a missing image or crop")
            if file_sha(image) != row["image_sha256"]:
                raise ValueError("Source image changed since crop generation")
            if row["source_label"] not in labels:
                raise ValueError("Unexpected person class in review manifest")
            if row["review_status"] != "pending" or row["reviewed_label"]:
                raise ValueError("Review manifest contains reviewed or changed labels; explicitly incorporate them first")
            prepared.append(
                {
                    "path": crop.relative_to(ROOT).as_posix(),
                    "source_image": image.relative_to(ROOT).as_posix(),
                    "source_image_sha256": row["image_sha256"],
                    "label": labels[row["source_label"]],
                    "label_status": "source_unreviewed",
                    "group_id": ids[groups.find(row["image"])],
                }
            )
    by_group = defaultdict(set)
    for row in prepared:
        by_group[row["group_id"]].add(row["label"])
    conflict = {key for key, values in by_group.items() if len(values) > 1}
    kept = []
    for row in prepared:
        if row["group_id"] in conflict:
            excluded.append({"path": row["path"], "reason": "candidate_group_has_conflicting_labels"})
        else:
            kept.append(row)
    return (
        kept,
        excluded,
        {
            "source_dataset": "smoking_people_roboflow_v2",
            "source_review_csv_sha256": file_sha(source_csv),
            "candidate_near_duplicate_pairs": candidate_pairs,
            "source_review_rows_without_crops_omitted": dict(omitted_review_rows),
            "conflicting_label_groups_excluded": len(conflict),
            "grouping": "same source image, normalized filename or transitive 64-bit dHash distance <= 3",
            "label_limitation": "All labels are unreviewed source annotations; test labels require independent review.",
            "independence_limitation": (
                "Source camera, subject and recording identities are unknown. Similarity grouping is incomplete."
            ),
            "rights_limitation": "Repository lists CC BY 4.0; underlying image rights are unverified.",
        },
    )


def video_hashes(path):
    """Decode three frames to screen for visually similar clips; not exhaustive."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or frames < 1:
            return None
        result = []
        for fraction in (0.2, 0.5, 0.8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frames - 1, int((frames - 1) * fraction)))
            ok, frame = cap.read()
            if not ok:
                return None
            gray = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = gray.tobytes()
            value = 0
            for y in range(8):
                for x in range(8):
                    value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            result.append(value)
        return tuple(result)
    finally:
        cap.release()


def violence_records():
    """Exclude ambiguous provenance and group clips by recording and similar sampled frames."""
    dataset = ROOT / "data/violence/fight-detection-surv-dataset"
    sources = defaultdict(list)
    current = None
    index = dataset / "videos.txt"
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(("https://", "http://")):
            current = line
        elif match := ENTRY.match(line):
            sources[match.group(1).lower()].append(current)
    clips, excluded = [], []
    for folder, label in (("fight", "violent"), ("noFight", "non_violent")):
        for path in sorted((dataset / folder).glob("*.mp4")):
            found = sources[path.stem.lower()]
            if not found or len(found) != 1 or found[0] is None:
                excluded.append({"path": path.relative_to(ROOT).as_posix(), "reason": "missing_or_ambiguous_source"})
                continue
            fingerprint = video_hashes(path)
            if fingerprint is None:
                excluded.append({"path": path.relative_to(ROOT).as_posix(), "reason": "unreadable_sample_frames"})
                continue
            clips.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "label": label,
                    "source": found[0],
                    "sha256": file_sha(path),
                    "fingerprint": fingerprint,
                }
            )
    groups = Groups(sorted({row["source"] for row in clips}))
    candidate_pairs = []
    for i, row in enumerate(clips):
        for other in clips[:i]:
            if row["source"] == other["source"]:
                continue
            is_exact = row["sha256"] == other["sha256"]
            is_similar = all((a ^ b).bit_count() <= 3 for a, b in zip(row["fingerprint"], other["fingerprint"], strict=True))
            if is_exact or is_similar:
                groups.union(row["source"], other["source"])
                candidate_pairs.append({"left": row["path"], "right": other["path"], "exact": is_exact})
    members = defaultdict(list)
    for source in groups.parent:
        members[groups.find(source)].append(source)
    ids = {root: group_identifier(items) for root, items in members.items()}
    rows = [
        {
            "path": row["path"],
            "label": row["label"],
            "label_status": "source_unreviewed",
            "source_recording": row["source"],
            "group_id": ids[groups.find(row["source"])],
            "sha256": row["sha256"],
        }
        for row in clips
    ]
    return (
        rows,
        excluded,
        {
            "source_dataset": "surveillance_camera_fight_ff83b7c521e5ca4eb67a212cc42dbc54f39d6290",
            "source_index_sha256": file_sha(index),
            "grouping": "source recording URL, plus transitive exact or three-sampled-frame dHash matches",
            "cross_recording_candidate_pairs": candidate_pairs,
            "label_limitation": "Clip-level source labels are unreviewed; timestamps and hard negatives need audit.",
            "independence_limitation": (
                "Some clips lack source mapping and were excluded. Three-frame screening cannot detect all duplicates."
            ),
            "rights_limitation": "Repository lists MIT; rights to underlying videos and source datasets remain unverified.",
        },
    )


def write_splits(task, rows, excluded, metadata, output, seed):
    if not rows:
        raise ValueError("No usable examples after exclusions")
    mapping = choose_group_split(rows, seed=seed)
    summary = {}
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        selected = sorted((row for row in rows if mapping[row["group_id"]] == split), key=lambda row: row["path"])
        with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
        summary[split] = {
            "rows": len(selected),
            "labels": dict(Counter(row["label"] for row in selected)),
            "groups": len({row["group_id"] for row in selected}),
        }
    with (output / "excluded.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(excluded)
    metadata.update(
        {
            "task": task,
            "seed": seed,
            "split_targets": dict(zip(SPLITS, FRACTIONS, strict=True)),
            "status": "provisional_source_labels_unreviewed",
            "summary": summary,
            "excluded": len(excluded),
        }
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("smoking", "violence"))
    parser.add_argument("output", type=Path, help="New directory under data/splits/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = args.output.resolve()
    permitted = (ROOT / "data/splits").resolve()
    if not target.is_relative_to(permitted) or target == permitted or target.exists():
        parser.error("output must be a new directory under data/splits/")
    print(f"Grouping {args.task} sources; no model training will occur.", flush=True)
    rows, excluded, metadata = smoking_records() if args.task == "smoking" else violence_records()
    result = write_splits(args.task, rows, excluded, metadata, target, args.seed)
    print(json.dumps({"summary": result["summary"], "excluded": result["excluded"]}, indent=2))


if __name__ == "__main__":
    main()
