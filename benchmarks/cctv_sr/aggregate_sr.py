"""Aggregate the per-run JSON files in results/cctv_sr/sr/ into summary tables.

Writes:
    results/cctv_sr/sr_summary.csv  mean +/- 95% CI over folds, per (dataset, task, model)
    results/cctv_sr/sr_paired.csv   paired per-fold differences between models

Fairness rule: for each (dataset, task) only folds where ALL THREE models finished are used, so every model is
averaged over the same folds. This matters because the benchmark was stopped early and some folds are missing.
Intervals are across folds (the independent units), not across images or seeds. A CI needs >= 3 folds.

Usage: python benchmarks/cctv_sr/aggregate_sr.py
"""

import glob
import json

import numpy as np
import pandas as pd
from common import MODELS, RESULTS
from scipy import stats

METRICS = ["map50", "map50_95", "class_acc_matched", "train_min"]
# (A, B): reported as A minus B
PAIRS = [("yolo26s", "yolo26n"), ("rtdetr-l", "yolo26n"), ("rtdetr-l", "yolo26s")]


def mean_ci(values) -> tuple[float, float]:
    """Mean and 95% t-confidence half-width (NaN with fewer than 3 values)."""
    v = np.asarray(values, float)
    n = len(v)
    half_width = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan
    return v.mean(), half_width


def main() -> None:
    runs = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(RESULTS / "sr/*.json"))])
    summary_rows, paired_rows = [], []

    for (dataset, task), group in runs.groupby(["dataset", "task"]):
        # one row per fold, one column per model; keep folds where all three models have a result
        by_fold = group.pivot(index="fold", columns="model", values="map50_95").reindex(columns=list(MODELS))
        folds = by_fold.dropna().index.tolist()

        for model in MODELS:
            rows = group[(group.model == model) & group.fold.isin(folds)].sort_values("fold")
            row = {
                "dataset": dataset,
                "task": task,
                "model": model,
                "folds_used": len(folds),
                "folds": ",".join(map(str, folds)),
            }
            for metric in METRICS:
                if metric in rows and rows[metric].notna().any():
                    row[metric], row[metric + "_ci"] = mean_ci(rows[metric].dropna())
            summary_rows.append(row)

        # paired comparison: per-fold difference between two models, then mean +/- CI of those differences
        for a, b in PAIRS:
            if len(folds) >= 3:
                diff = (by_fold.loc[folds, a] - by_fold.loc[folds, b]).values
                mean, ci = mean_ci(diff)
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "A": a,
                        "B": b,
                        "diff": mean,
                        "ci": ci,
                        "folds_A_better": int((diff > 0).sum()),
                        "n_folds": len(folds),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(RESULTS / "sr_summary.csv", index=False)
    paired.to_csv(RESULTS / "sr_paired.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.round(3).to_string())
    print(paired.round(3).to_string())
    print(runs.groupby(["dataset", "task", "model"]).size().unstack().to_string())  # runs per cell


if __name__ == "__main__":
    main()
