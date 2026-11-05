import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/aitf_cv/tasks/smoking/audit_yolo_boxes.py"
spec = importlib.util.spec_from_file_location("smoking_yolo_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def dataset(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [cigarette, non smoking person, smoking person]\n")
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
    return tmp_path


def test_audit_reports_boxes_missing_labels_and_cross_split_duplicates(tmp_path):
    root = dataset(tmp_path)
    (root / "train/images/a.jpg").write_bytes(b"shared")
    (root / "train/labels/a.txt").write_text("2 0.5 0.5 0.5 0.5\n")
    (root / "valid/images/b.jpg").write_bytes(b"other")
    (root / "valid/labels/b.txt").write_text("1 0.9 0.5 0.5 0.5\n")
    (root / "test/images/c.jpg").write_bytes(b"shared")
    result = module.audit(root)
    assert result["box_counts"]["train"]["smoking person"] == 1
    assert result["cross_split_duplicate_hash_groups"] == 1
    assert result["missing_label_files"] == ["test/images/c.jpg"]
    assert len(result["invalid_labels"]) == 1


def test_unknown_class_rejected(tmp_path):
    root = dataset(tmp_path)
    (root / "data.yaml").write_text("names: [person, cigarette]\n")
    with pytest.raises(ValueError, match="expected"):
        module.audit(root)


def test_annotated_crop_proposals_remain_unreviewed(tmp_path, monkeypatch):
    import csv
    import sys

    from PIL import Image

    (tmp_path / "source").mkdir()
    root = dataset(tmp_path / "source")
    for split in ("train", "valid", "test"):
        Image.new("RGB", (20, 10), "white").save(root / split / "images/a.jpg")
        (root / split / "labels/a.txt").write_text("2 0.5 0.5 0.5 0.6\n")
    monkeypatch.setitem(sys.modules, "audit_yolo_boxes", module)
    crop_spec = importlib.util.spec_from_file_location("smoking_annotated_crops", SCRIPT.parent / "propose_annotated_crops.py")
    cropper = importlib.util.module_from_spec(crop_spec)
    crop_spec.loader.exec_module(cropper)
    monkeypatch.setattr(cropper, "ROOT", tmp_path)
    target = tmp_path / "data/smoking/proposals"
    monkeypatch.setattr(sys, "argv", ["propose_annotated_crops", str(root), str(target)])
    cropper.main()
    with (target / "review.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(row["source_label"] == "smoking person" and not row["reviewed_label"] for row in rows)
    assert all(row["review_status"] == "pending" and (target / row["crop"]).is_file() for row in rows)
