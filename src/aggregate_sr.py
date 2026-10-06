import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
R = Path(__file__).resolve().parents[1] / "results"
d = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R / "sr/*.json"))])
M = ["yolo26n", "yolo26s", "rtdetr-l"]
def ci(v):
    v = np.asarray(v, float); n = len(v)
    return v.mean(), (stats.t.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan)
rows, pr = [], []
for (ds, t), x in d.groupby(["dataset", "task"]):
    P = x.pivot(index="fold", columns="model", values="map50_95").reindex(columns=M); full = P.dropna().index.tolist()  # folds where all 3 models finished
    for m in M:
        y = x[(x.model == m) & x.fold.isin(full)].sort_values("fold")
        r = {"dataset": ds, "task": t, "model": m, "folds_used": len(full), "folds": ",".join(map(str, full))}
        for c in ["map50", "map50_95", "class_acc_matched", "train_min"]:
            if c in y and y[c].notna().any(): r[c], r[c + "_ci"] = ci(y[c].dropna())
        rows.append(r)
    for a, b in [("yolo26s", "yolo26n"), ("rtdetr-l", "yolo26n"), ("rtdetr-l", "yolo26s")]:
        if len(full) >= 3:
            dif = (P.loc[full, a] - P.loc[full, b]).values; m_, c_ = ci(dif); pr.append({"dataset": ds, "task": t, "A": a, "B": b, "diff": m_, "ci": c_, "folds_A_better": int((dif > 0).sum()), "n_folds": len(full)})
s = pd.DataFrame(rows); s.to_csv(R / "sr_summary.csv", index=False); p = pd.DataFrame(pr); p.to_csv(R / "sr_paired.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(s.round(3).to_string()); print(p.round(3).to_string()); print(d.groupby(["dataset", "task", "model"]).size().unstack().to_string())
