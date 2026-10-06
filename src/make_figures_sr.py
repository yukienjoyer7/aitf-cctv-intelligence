import json, glob, re
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; O = R / "fig_sr"; O.mkdir(exist_ok=True)
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"; BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8e8e93"
MC = {"yolo26n": BLUE, "yolo26s": ORANGE, "rtdetr-l": AQUA}; MN = {"yolo26n": "YOLO26n", "yolo26s": "YOLO26s", "rtdetr-l": "RT-DETR-L"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
def lab(ds, split, im):
    names = yaml.safe_load((ROOT / "data" / ds / "data.yaml").read_text())["names"]; out = []
    for ln in (ROOT / "data" / ds / split / "labels" / (Path(im).stem + ".txt")).read_text().splitlines():
        c, *v = ln.split(); v = list(map(float, v))
        if len(v) > 4: xs, ys = v[0::2], v[1::2]; v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
        out.append((names[int(c)], v))
    return out
def col(n): return ORANGE if n.startswith("Female") else BLUE if n.startswith("Male") else GRAY
def example(ds, pick, title, fn):
    cands = sorted((ROOT / "data" / ds / "train" / "images").glob("*.jpg")); best = None
    for p in cands[::3]:
        L = lab(ds, "train", p); ppl = sum(not n.startswith(("Window", "Door")) for n, _ in L)
        if pick(ppl, L): best = (p, L); break
    p, L = best; im = Image.open(p).convert("RGB"); fig, ax = plt.subplots(figsize=(7, 7)); ax.imshow(im); ax.axis("off")
    for n, (x, y, w, h) in L:
        ax.add_patch(Rectangle(((x - w / 2) * 512, (y - h / 2) * 512), w * 512, h * 512, fill=False, ec=col(n), lw=1.6, ls="--" if col(n) == GRAY else "-"))
    ax.set_title(title, loc="left", fontsize=13)
    for i, (t, c) in enumerate([("Female", ORANGE), ("Male", BLUE), ("Window / door (static)", GRAY)]):
        fig.text(0.04 + [0, .17, .30][i], 0.03, "■", color=c, fontsize=13); fig.text(0.07 + [0, .17, .30][i], 0.032, t, color=INK2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(O / fn, dpi=200); plt.close(fig)
example("indoor", lambda n, L: n >= 14, "Indoor: classroom, one fixed camera", "sr_01_indoor_example.png")
example("outdoor", lambda n, L: n >= 4, "Outdoor: corridor, one fixed camera", "sr_02_outdoor_example.png")
# class counts
cnt = {}
for ds in ["indoor", "outdoor"]:
    names = yaml.safe_load((ROOT / "data" / ds / "data.yaml").read_text())["names"]; c = dict.fromkeys(names, 0)
    for f in (ROOT / "data" / ds).glob("*/labels/*.txt"):
        for ln in f.read_text().splitlines(): c[names[int(ln.split()[0])]] += 1
    cnt[ds] = c
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [3, 9]})
for ax, ds in zip(axs, ["indoor", "outdoor"]):
    c = cnt[ds]; ks = list(c)[::-1]; ax.barh(ks, [c[k] for k in ks], color=[col(k) for k in ks], height=0.62, zorder=2); ax.set_xscale("log"); ax.grid(axis="x", color=GRID, zorder=0)
    for k in ks: ax.text(c[k] * 1.12, k, f"{c[k]:,}", va="center", fontsize=9.5, color=INK2)
    ax.set_xlim(1, max(c.values()) * 8); ax.set_title(f"{ds.capitalize()} ({sum(c.values()):,} boxes)", loc="left"); ax.set_xlabel("boxes (log scale)")
fig.suptitle("Class balance: windows/doors (gray) dominate, especially outdoors", x=0.01, ha="left", fontsize=13); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(O / "sr_03_class_counts.png", dpi=200); plt.close(fig)
# results
d = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R / "sr/*.json"))]); s = pd.read_csv(R / "sr_summary.csv")
def results(ds, tasks, fn, ttl):
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks) + 0.6, 4.4), sharey=True)
    for ax, t in zip(axs, tasks):
        sm = s[(s.dataset == ds) & (s.task == t)]; nf = int(sm.folds_used.iloc[0]); full = [int(x) for x in sm.folds.iloc[0].split(",")]
        for i, m in enumerate(MC):
            r = sm[sm.model == m].iloc[0]; ys = d[(d.dataset == ds) & (d.task == t) & (d.model == m) & d.fold.isin(full)].sort_values("fold").map50_95
            ax.bar(i, r.map50_95, width=.6, color=MC[m], yerr=r.map50_95_ci, capsize=4, error_kw=dict(ecolor=INK2, lw=1), zorder=2)
            ax.scatter(np.full(len(ys), i) + np.linspace(-.14, .14, len(ys)), ys, s=14, color="white", ec=INK2, lw=.8, zorder=3)
            ax.text(i, 0.02, f"{r.map50_95:.2f}", ha="center", color="white", fontsize=11, fontweight="bold", zorder=4)
        ax.set_xticks(range(3), [MN[m] for m in MC], rotation=15); ax.set_title(f"{t} ({nf} folds)", loc="left", fontsize=11.5); ax.grid(axis="y", color=GRID, zorder=0); ax.set_ylim(0, 0.9)
    axs[0].set_ylabel("mAP50-95"); fig.suptitle(ttl, x=0.01, ha="left", fontsize=12.5); fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(O / fn, dpi=200); plt.close(fig)
results("indoor", ["person", "gender", "allclass"], "sr_04_indoor_results.png", "Indoor: mAP50-95, bar = mean +/- 95% CI over folds, dots = folds (seed 0)")
results("outdoor", ["person", "gender", "posture", "allclass"], "sr_05_outdoor_results.png", "Outdoor: mAP50-95, bar = mean +/- 95% CI over folds, dots = folds (seed 0)")
# outdoor fold density
rows = []
for k in range(5):
    t = sum(1 for f in (ROOT / "data/sr/outdoor/person" / f"fold{k}/test/labels").glob("*.txt") for _ in f.read_text().splitlines())
    n = len(list((ROOT / "data/sr/outdoor/person" / f"fold{k}/test/labels").glob("*.txt"))); rows.append((k, t, n))
pm = d[(d.dataset == "outdoor") & (d.task == "person")].groupby(["fold", "model"]).map50_95.mean().unstack()
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
axs[0].bar([f"Fold {k}" for k, _, _ in rows], [t for _, t, _ in rows], color=GRAY, width=.6, zorder=2); axs[0].grid(axis="y", color=GRID, zorder=0)
for i, (_, t, n) in enumerate(rows): axs[0].text(i, t + 40, f"{t:,}", ha="center", fontsize=10, color=INK2)
axs[0].set_title("People boxes in each held-out block (~187 frames)", loc="left", fontsize=11); axs[0].set_ylabel("people boxes")
for m in MC: axs[1].plot(range(5), pm[m].reindex(range(5)), marker="o", color=MC[m], label=MN[m], lw=1.6)
axs[1].set_xticks(range(5), [f"Fold {k}" for k in range(5)]); axs[1].set_ylabel("person mAP50-95"); axs[1].grid(axis="y", color=GRID); axs[1].legend(frameon=False, fontsize=9); axs[1].set_title("Person mAP50-95 by fold", loc="left", fontsize=11)
fig.suptitle("Outdoor folds are very different: people are sparse early and crowded late", x=0.01, ha="left", fontsize=12.5); fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(O / "sr_06_outdoor_fold_density.png", dpi=200); plt.close(fig)
print(rows); print(pm.round(3).to_string())
