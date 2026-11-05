import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
prepare = importlib.import_module("aitf_cv.tasks.prepare_splits")
utils = importlib.import_module("aitf_cv.tasks.split_utils")


def test_group_split_is_deterministic_and_never_splits_group():
    rows = [{"group_id": f"group-{i}", "label": label} for i in range(20) for label in ("one", "two")]
    first = utils.choose_group_split(rows, seed=42, trials=200)
    assert first == utils.choose_group_split(rows, seed=42, trials=200)
    assert set(first.values()) == {"train", "val", "test"}
    assert all(first[row["group_id"]] in utils.SPLITS for row in rows)
    with pytest.raises(ValueError):
        utils.choose_group_split([{"group_id": "only", "label": "one"}])


def test_smoking_near_duplicate_sources_stay_grouped_and_conflicts_are_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    dataset = tmp_path / "data/smoking/smoking_people_v2"
    review = tmp_path / "data/smoking/smoking_people_review_v1"
    (review / "crops").mkdir(parents=True)
    cases = [
        ("train", "a", False, "smoking person"),
        ("test", "b", False, "smoking person"),
        ("valid", "c", True, "non smoking person"),
    ]
    rows = []
    for split, name, reverse, label in cases:
        image = dataset / split / "images" / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        gray = Image.new("L", (9, 8))
        gray.putdata([((8 - x if reverse else x) * 25) for _ in range(8) for x in range(9)])
        gray.save(image)
        crop = review / "crops" / f"{name}.png"
        gray.save(crop)
        rows.append(
            {
                "split": split,
                "image": image.relative_to(dataset).as_posix(),
                "image_sha256": prepare.file_sha(image),
                "crop": crop.relative_to(review).as_posix(),
                "source_label": label,
                "reviewed_label": "",
                "review_status": "pending",
                "xyxy": "",
            }
        )
    with (review / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records, excluded, _ = prepare.smoking_records()
    assert not excluded
    assert records[0]["group_id"] == records[1]["group_id"]
    assert records[0]["group_id"] != records[2]["group_id"]
    assert {row["label_status"] for row in records} == {"source_unreviewed"}
    rows[1]["source_label"] = "non smoking person"
    with (review / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records, excluded, _ = prepare.smoking_records()
    assert len(excluded) == 2
    assert all(row["reason"] == "candidate_group_has_conflicting_labels" for row in excluded)


def test_violence_excludes_unmapped_and_keeps_recordings_together(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "video_hashes", lambda path: (0, 0, 0) if path.stem in {"fi001", "fi002"} else (7, 7, 7))
    root = tmp_path / "data/violence/fight-detection-surv-dataset"
    for folder in ("fight", "noFight"):
        (root / folder).mkdir(parents=True)
    for name in ("fi001", "fi002", "fi003"):
        (root / "fight" / f"{name}.mp4").write_bytes(name.encode())
    (root / "noFight/nofi001.mp4").write_bytes(b"unmapped")
    (root / "videos.txt").write_text("https://example/a\nfi001: 0 - 2\nfi002: 2 - 4\nhttps://example/b\nfi003: 0 - 2\n")
    rows, excluded, _ = prepare.violence_records()
    grouped = defaultdict(set)
    for row in rows:
        grouped[row["source_recording"]].add(row["group_id"])
    assert len(grouped["https://example/a"]) == 1
    assert excluded[0]["reason"] == "missing_or_ambiguous_source"
    assert all(row["label_status"] == "source_unreviewed" for row in rows)
