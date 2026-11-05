import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src/aitf_cv/tasks/smoking/audit_images.py"
spec = importlib.util.spec_from_file_location("smoking_image_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_detects_cross_split_duplicates_and_class_conflicts(tmp_path):
    for folder in ("training_data/smoking", "validation_data/notsmoking", "testing_data"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "training_data/smoking/one.jpg").write_bytes(b"same")
    (tmp_path / "validation_data/notsmoking/two.jpg").write_bytes(b"same")
    (tmp_path / "testing_data/three.jpg").write_bytes(b"different")
    result = module.audit(tmp_path)
    assert result["images"] == 3
    assert result["unique_sha256"] == 2
    assert result["cross_supplied_folder_duplicate_groups"] == 1
    assert result["conflicting_supplied_class_groups"] == 1
