import json, glob, numpy as np, pandas as pd
from scipy import stats
rows = []
for f in glob.glob("results/ir/*.json"):
    j = json.load(open(f))
    for k, v in j["results"].items():
        tr, te = k.replace("train_", "").split("__test_")
        rows.append({"task": j["task"], "fold": j["fold"], "seed": j["seed"], "train": tr, "test": te, **v})
d = pd.DataFrame(rows); g = d.groupby(["task", "train", "test", "fold"])[["map50", "map50_95"]].mean().reset_index()
def ci(v): v = np.asarray(v); return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
out = []
for (t, tr, te), x in g.groupby(["task", "train", "test"]):
    m, c = ci(x.map50_95); m5, c5 = ci(x.map50); out.append({"task": t, "train": tr, "test": te, "map50": m5, "map50_ci": c5, "map50_95": m, "ci": c})
s = pd.DataFrame(out); s.to_csv("results/ir_summary.csv", index=False)
pd.set_option("display.width", 200)
for t, x in s.groupby("task"):
    print("\n", t, "mAP50-95 (rows=train, cols=test)"); print(x.pivot(index="train", columns="test", values="map50_95").round(3).loc[["rgb", "gray", "irsim"], ["rgb", "gray", "irsim"]])
    print("mAP50"); print(x.pivot(index="train", columns="test", values="map50").round(3).loc[["rgb", "gray", "irsim"], ["rgb", "gray", "irsim"]])
pairs = [("rgb", "rgb", "rgb", "gray"), ("rgb", "rgb", "rgb", "irsim"), ("gray", "gray", "rgb", "gray"), ("irsim", "irsim", "rgb", "irsim"),
         ("irsim", "irsim", "rgb", "rgb"), ("gray", "gray", "rgb", "rgb"), ("irsim", "irsim", "gray", "gray")]
print("\npaired diffs mAP50-95 (A minus B), n=5 folds")
pr = []
for t, x in g.groupby("task"):
    P = x.set_index(["train", "test", "fold"]).map50_95
    for a1, a2, b1, b2 in pairs:
        dif = np.array([P[(a1, a2, f)] - P[(b1, b2, f)] for f in range(5)]); m, c = ci(dif)
        pr.append({"task": t, "A": f"train {a1} test {a2}", "B": f"train {b1} test {b2}", "diff": m, "ci": c, "folds_positive": int((dif > 0).sum())})
pr = pd.DataFrame(pr); pr.to_csv("results/ir_paired.csv", index=False); print(pr.round(3).to_string())
print(d.groupby("seed").size().to_dict(), "rows;", d.groupby(["task","fold"]).seed.nunique().min(), "min seeds/fold")
