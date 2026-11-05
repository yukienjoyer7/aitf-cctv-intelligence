import importlib.util
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "src/aitf_cv/tasks/smoking/audit_near_duplicates.py"
spec = importlib.util.spec_from_file_location("smoking_near_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_candidate_across_splits_and_comparison(tmp_path):
    root = tmp_path / "source"
    for split in ("train", "test"):
        (root / split).mkdir(parents=True)
        Image.new("RGB", (16, 16), "red").save(root / split / "sample.png")
    other = tmp_path / "other"
    other.mkdir()
    Image.new("RGB", (16, 16), "red").save(other / "sample.png")
    result = module.audit(root, other, distance=0)
    assert result["cross_split_candidate_pairs"] == 1
    assert result["comparison_candidate_pairs"] == 2
    assert result["primary_images"] == 2
