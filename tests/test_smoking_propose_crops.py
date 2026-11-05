import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src/aitf_cv/tasks/smoking/propose_crops.py"
spec = importlib.util.spec_from_file_location("smoking_crops", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_crop_bounds():
    assert module.clipped_box([-1.2, 1.2, 9.1, 12], 10, 10) == (0, 1, 10, 10)
    assert module.clipped_box([11, 1, 12, 3], 10, 10) is None
    assert module.clipped_box([1, 1, 1, 2], 10, 10) is None


def test_missing_inputs_fail_without_model_import(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "missing"), str(tmp_path / "missing.pt"), str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "images directory and local weights file must exist" in result.stderr
    assert not (tmp_path / "out").exists()


def test_proposals_require_manual_labels(tmp_path, monkeypatch):
    import csv
    import json
    import types

    from PIL import Image

    class Vector:
        def tolist(self):
            return [1, 1, 6, 6]

    class Box:
        xyxy = [Vector()]
        conf = [0.75]

    class Detector:
        names = {0: "person"}

        def predict(self, *args, **kwargs):
            return [types.SimpleNamespace(boxes=[Box()])]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: Detector()))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10)).save(images / "example.jpg")
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"checkpoint")
    target = tmp_path / "data/smoking/review"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(images), str(weights), str(target)])
    module.main()
    with (target / "review.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["reviewed_label"] == ""
    assert row["review_status"] == "pending"
    assert (target / row["crop"]).is_file()
    assert json.loads((target / "metadata.json").read_text())["confidence_status"] == "uncalibrated"
