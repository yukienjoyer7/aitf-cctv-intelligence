import importlib.util
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
SCRIPT = Path(__file__).resolve().parents[1] / "src/aitf_cv/tasks/smoking/audit_hf_parquet.py"
spec = importlib.util.spec_from_file_location("smoking_hf_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_identical_images_across_splits_and_comparison(tmp_path):
    parquet = tmp_path / "parquet"
    for split, label in (("train", 0), ("test", 1)):
        (parquet / split).mkdir(parents=True)
        table = pa.Table.from_pylist([{"image": {"bytes": b"same", "path": f"{split}.jpg"}, "label": label}])
        pq.write_table(table, parquet / split / "0000.parquet")
    comparison = tmp_path / "old"
    comparison.mkdir()
    (comparison / "old.jpg").write_bytes(b"same")
    result = module.audit(parquet, comparison)
    assert result["cross_train_test_duplicate_hash_groups"] == 1
    assert result["conflicting_label_hash_groups"] == 1
    assert result["overlap_with_comparison_rows"] == {"train": 1, "test": 1}
    assert result["counts"]["test"]["smoking"] == 1


def test_missing_split_fails(tmp_path):
    with pytest.raises(ValueError, match="No completed Parquet"):
        module.audit(tmp_path)


def test_extracts_labels_without_overwriting(tmp_path):
    extract_script = SCRIPT.parent / "extract_hf_parquet.py"
    extract_spec = importlib.util.spec_from_file_location("extract_smoking_hf", extract_script)
    extract_module = importlib.util.module_from_spec(extract_spec)
    extract_spec.loader.exec_module(extract_module)
    parquet = tmp_path / "parquet"
    for split, label in (("train", 0), ("test", 1)):
        (parquet / split).mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist([{"image": {"bytes": b"image", "path": "picture.jpg"}, "label": label}]),
            parquet / split / "0000.parquet",
        )
    output = tmp_path / "images"
    assert extract_module.extract(parquet, output) == 2
    assert (output / "train/not_smoking/picture.jpg").read_bytes() == b"image"
    assert (output / "test/smoking/picture.jpg").read_bytes() == b"image"
    with pytest.raises(FileExistsError):
        extract_module.extract(parquet, output)
