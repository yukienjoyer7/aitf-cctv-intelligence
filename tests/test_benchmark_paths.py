import importlib
import subprocess
import sys
from pathlib import Path

import common
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_paths():
    assert common.ROOT == ROOT
    assert common.DATA == ROOT / "data"
    assert common.RESULTS == ROOT / "results" / "cctv_sr"


def test_eval_help_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/cctv_sr/sr_eval.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--dataset" in result.stdout and "--epochs" in result.stdout


def test_configs_and_package():
    sys.path.insert(0, str(ROOT / "src"))
    importlib.import_module("aitf_cv.tasks")
    for path in (ROOT / "configs/tasks").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert config["status"] == ("benchmark_only" if path.stem == "person_attributes" else "planned")
        assert config["training_data"] == []
    assert yaml.safe_load((ROOT / "configs/datasets/registry.example.yaml").read_text())["license_review_status"] == "pending"
