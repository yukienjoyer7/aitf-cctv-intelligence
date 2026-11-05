import json
import shutil
from pathlib import Path

import aggregate_sr
import pandas as pd
import pandas.testing as pdt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/cctv_sr"


def test_runs_and_missing_folds():
    runs = [json.loads(path.read_text()) for path in (RESULTS / "sr").glob("*.json")]
    assert len(runs) == 96
    assert {r["seed"] for r in runs} == {0}
    for task in ("gender", "posture", "allclass"):
        assert {r["fold"] for r in runs if r["dataset"] == "outdoor" and r["task"] == task} == {0, 1, 2, 3}


def test_aggregation_matches_archived_tables(tmp_path, monkeypatch, capsys):
    shutil.copytree(RESULTS / "sr", tmp_path / "sr")
    monkeypatch.setattr(aggregate_sr, "RESULTS", tmp_path)
    aggregate_sr.main()
    capsys.readouterr()
    for name in ("sr_summary.csv", "sr_paired.csv"):
        actual = pd.read_csv(tmp_path / name)
        expected = pd.read_csv(RESULTS / name)
        pdt.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-12)
