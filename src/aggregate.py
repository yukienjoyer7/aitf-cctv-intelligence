"""Aggregate results -> results/summary.csv, LaTeX tables, figures. Unit of analysis = fold (seed-averaged)."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"; (R / "fig").mkdir(exist_ok=True)
d = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R / "runs/*.json"))])
NAMES = {"yolo26n": "YOLO26n", "yolo26s": "YOLO26s", "rtdetr-l": "RT-DETR-L"}
ORDER = list(NAMES)
g = d.groupby(["task", "model", "fold"])[["map50", "map50_95", "gender_acc_matched", "train_min"]].mean().reset_index()

def ci(v):
    v = np.asarray(v.dropna()); return (v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (np.nan, np.nan)
rows = []
for (t, m), x in g.groupby(["task", "model"]):
    r = {"task": t, "model": m, "n_runs": int(((d.task == t) & (d.model == m)).sum())}
    for c in ["map50", "map50_95", "gender_acc_matched", "train_min"]:
        r[c], r[c + "_ci"] = ci(x[c])
    rows.append(r)
s = pd.DataFrame(rows); s.to_csv(R / "summary.csv", index=False)

# paired per-fold differences vs YOLO26n (5 folds -> report mean diff and 95% CI; no p-values claimed)
pair = []
for t, x in g.groupby("task"):
    p = x.pivot(index="fold", columns="model", values="map50_95")
    for a, b in [("yolo26s", "yolo26n"), ("rtdetr-l", "yolo26n"), ("rtdetr-l", "yolo26s")]:
        dm, dc = ci(p[a] - p[b]); pair.append({"task": t, "a": a, "b": b, "diff": dm, "ci": dc})
pd.DataFrame(pair).to_csv(R / "paired_diffs.csv", index=False)

def f(m, c): return f"{m:.3f} $\\pm$ {c:.3f}"
lines = ["\\begin{tabular}{llcccc}\\toprule", "Task & Model & mAP$_{50}$ & mAP$_{50\\text{-}95}$ & Gender acc. & Train (min) \\\\ \\midrule"]
for t in ["person", "gender"]:
    for m in ORDER:
        r = s[(s.task == t) & (s.model == m)].iloc[0]
        ga = f(r.gender_acc_matched, r.gender_acc_matched_ci) if t == "gender" else "--"
        lines.append(f"{t.capitalize()} & {NAMES[m]} & {f(r.map50, r.map50_ci)} & {f(r.map50_95, r.map50_95_ci)} & {ga} & {r.train_min:.1f} \\\\")
lines += ["\\bottomrule\\end{tabular}"]
(R / "table_accuracy.tex").write_text("\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for a, t in zip(ax, ["person", "gender"]):
    x = s[s.task == t].set_index("model").loc[ORDER]
    a.bar([NAMES[m] for m in ORDER], x.map50_95, yerr=x.map50_95_ci, capsize=4, color=["#9ecae1", "#4292c6", "#08519c"])
    a.set_title(f"{t.capitalize()} detection"); a.set_ylim(0.6, 0.85); a.tick_params(axis="x", rotation=15)
ax[0].set_ylabel("mAP$_{50\\text{-}95}$ (mean, 95% CI over folds)"); fig.tight_layout(); fig.savefig(R / "fig/map_bars.pdf")

lp = R / "latency.json"
if lp.exists():
    L = json.load(open(lp))["models"]
    lines = ["\\begin{tabular}{lrrrrrr}\\toprule", "Model & Params (M) & GFLOPs & GPU fp32 (ms) & GPU fp16 (ms) & CPU (ms) & FPS (fp16 e2e) \\\\ \\midrule"]
    for m in ORDER:
        z = L[m]; e = lambda k: z[k]["end_to_end_ms"]["mean"]
        lines.append(f"{NAMES[m]} & {z['params_M']:.1f} & {z['gflops']:.1f} & {e('cuda_fp32'):.1f} & {e('cuda_fp16'):.1f} & {e('cpu_fp32'):.0f} & {z['cuda_fp16']['fps_e2e']:.0f} \\\\")
    lines += ["\\bottomrule\\end{tabular}"]; (R / "table_latency.tex").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for m in ORDER:
        for t, mk in [("person", "o"), ("gender", "s")]:
            r = s[(s.task == t) & (s.model == m)].iloc[0]
            ax.errorbar(L[m]["cuda_fp16"]["end_to_end_ms"]["mean"], r.map50_95, yerr=r.map50_95_ci, fmt=mk, capsize=3, label=f"{NAMES[m]} ({t})")
    ax.set_xlabel("End-to-end latency, GPU fp16 (ms)"); ax.set_ylabel("mAP$_{50\\text{-}95}$"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(R / "fig/pareto.pdf")
print(s.round(3).to_string()); print(pd.DataFrame(pair).round(3).to_string())
