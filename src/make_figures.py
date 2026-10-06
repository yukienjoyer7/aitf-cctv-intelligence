"""PNG figures for the slide deck -> results/fig_ppt/. Palette: reference dataviz palette (blue/orange/aqua = first 3 slots, validated all-pairs)."""
import json, glob, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; O = R / "fig_ppt"; O.mkdir(exist_ok=True)
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MC = {"yolo26n": BLUE, "yolo26s": ORANGE, "rtdetr-l": AQUA}; MN = {"yolo26n": "YOLO26n", "yolo26s": "YOLO26s", "rtdetr-l": "RT-DETR-L"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
FR = 30
rgb = Image.open(ROOT / "data/folds/gender/fold1/test/images" / f"f{FR:04d}.jpg").convert("RGB")
lab = [l.split() for l in (ROOT / "data/folds/gender/fold1/test/labels" / f"f{FR:04d}.txt").read_text().splitlines()]

# spec: load the same degradation functions used to build the datasets
sp = importlib.util.spec_from_file_location("prep_ir_fn", ROOT / "src/prep_ir.py")
def to_gray(im): return im.convert("L")
def steps(im, seed=FR):
    rng = np.random.default_rng(seed); g = np.asarray(im.convert("L"), np.float32) / 255; out = {"Gray": g.copy()}
    g = 0.15 + 0.7 * g; out["+ low contrast"] = g.copy()
    h, w = g.shape; yy, xx = np.mgrid[:h, :w]; r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    g = g * (1 - 0.35 * np.clip(r, 0, 1.2) ** 2); out["+ vignette"] = g.copy()
    b = np.asarray(Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255; out["+ blur"] = b.copy()
    out["+ sensor noise\n(= IR-sim)"] = np.clip(b + rng.normal(0, 0.04, b.shape), 0, 1)
    return out
S = steps(rgb)
irsim_img = Image.open(ROOT / "data/ir_images/irsim" / f"f{FR:04d}.jpg"); gray_img = Image.open(ROOT / "data/ir_images/gray" / f"f{FR:04d}.jpg")

def boxes(ax, size=512):
    for c, x, y, w, h in lab:
        x, y, w, h = [float(v) * size for v in (x, y, w, h)]
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, lw=1.6, ec=ORANGE if c == "0" else BLUE))

# 1 dataset example ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.9)); ax.imshow(rgb); boxes(ax); ax.axis("off")
ax.set_title("One annotated frame (512x512)", loc="left", fontsize=13)
ax.text(0, 1.0, "", transform=ax.transAxes)
fig.text(0.03, 0.02, "■", fontsize=13, color=ORANGE); fig.text(0.065, 0.022, "Female", fontsize=11, color=INK2)
fig.text(0.19, 0.02, "■", fontsize=13, color=BLUE); fig.text(0.225, 0.022, "Male", fontsize=11, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(O / "01_dataset_example.png", dpi=200); plt.close(fig)

# 2 blocked CV diagram ------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.4)); K, N, BUF = 5, 120, 4; bl = np.array_split(np.arange(N), K)
for k, b in enumerate(bl):
    y = K - 1 - k; lo, hi = b[0] - BUF, b[-1] + BUF
    for i in range(N):
        col = BLUE if b[0] <= i <= b[-1] else ("#d9d8d0" if lo <= i <= hi else "#a9c7ee")
        ax.add_patch(Rectangle((i, y + 0.12), 1, 0.76, fc=col, ec="none"))
    ax.text(-2, y + 0.5, f"Fold {k}", ha="right", va="center", color=INK2)
    ax.text(b[0] + 12, y + 0.5, "test (24)", ha="center", va="center", color="white", fontsize=9)
ax.set_xlim(-12, N); ax.set_ylim(-0.1, K + 0.9); ax.set_yticks([]); ax.set_xlabel("frame index (time order, ~1 s apart)"); ax.spines["left"].set_visible(False)
for x, t, c in [(0, "train", "#a9c7ee"), (16, "held-out test block", BLUE), (58, "4-frame buffer (dropped from train)", "#d9d8d0")]:
    ax.add_patch(Rectangle((x, K + 0.25), 3, 0.4, fc=c, ec="none")); ax.text(x + 4.5, K + 0.45, t, va="center", fontsize=10, color=INK2)
ax.set_title("Blocked 5-fold CV: each 24-frame block held out once", loc="left", fontsize=13)
fig.tight_layout(); fig.savefig(O / "02_blocked_cv.png", dpi=200); plt.close(fig)

# 3 IR examples ---------------------------------------------------------------
ir = pd.read_csv(R / "ir_summary.csv"); v = lambda t, tr, te: ir[(ir.task == t) & (ir.train == tr) & (ir.test == te)].map50_95.iloc[0]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.9))
for ax, im, name, te in zip(axs, [rgb, gray_img, irsim_img], ["RGB", "Gray", "IR-sim"], ["rgb", "gray", "irsim"]):
    ax.imshow(im); ax.axis("off"); ax.set_title(name, fontsize=14)
    ax.text(0.5, -0.05, f"RGB-trained model, person mAP50-95: {v('person', 'rgb', te):.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=INK2)
fig.suptitle("Simulated infrared: same frame, three conditions", x=0.01, ha="left", fontsize=14)
fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(O / "03_ir_examples.png", dpi=200); plt.close(fig)

# 4 degradation steps --------------------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(15, 3.6))
for ax, (k, g) in zip(axs, S.items()): ax.imshow(g, cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(k, fontsize=11)
fig.suptitle("How IR-sim is built (cumulative steps)", x=0.01, ha="left", fontsize=14); fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(O / "04_ir_degradation_steps.png", dpi=200); plt.close(fig)

# 5 heatmap (sequential single hue) ----------------------------------------------------
cm = LinearSegmentedColormap.from_list("b", ["#eaf1fb", "#9fc2ee", "#2a78d6", "#123e7a"])
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3)); C = ["rgb", "gray", "irsim"]; L = ["RGB", "Gray", "IR-sim"]
for ax, t in zip(axs, ["person", "gender"]):
    M = np.array([[v(t, a, b) for b in C] for a in C]); ax.imshow(M, cmap=cm, vmin=0, vmax=0.8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=13, color="white" if M[i, j] > 0.45 else INK, fontweight="bold" if i == j else None)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.4))
    ax.set_xticks(range(3), L); ax.set_yticks(range(3), L); ax.set_xlabel("Tested on"); ax.set_ylabel("Trained on"); ax.set_title(f"{t.capitalize()} detection", loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle("YOLO26n mAP50-95: train condition x test condition (mean of 5 folds)", x=0.01, ha="left", fontsize=13); fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(O / "05_ir_heatmap.png", dpi=200); plt.close(fig)

# 6 accuracy bars with per-fold dots --------------------------------------------------------
d = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R / "runs/*.json"))])
g = d.groupby(["task", "model", "fold"]).map50_95.mean().reset_index(); sm = pd.read_csv(R / "summary.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, t in zip(axs, ["person", "gender"]):
    for i, m in enumerate(MC):
        r = sm[(sm.task == t) & (sm.model == m)].iloc[0]; ys = g[(g.task == t) & (g.model == m)].map50_95
        ax.bar(i, r.map50_95, width=0.55, color=MC[m], yerr=r.map50_95_ci, capsize=4, error_kw=dict(ecolor=INK2, lw=1), zorder=2)
        ax.scatter(np.full(len(ys), i) + np.linspace(-.12, .12, len(ys)), ys, s=14, color="white", ec=INK2, lw=.8, zorder=3)
        ax.text(i, 0.615, f"{r.map50_95:.3f}", ha="center", color="white", fontsize=11, fontweight="bold", zorder=4)
    ax.set_xticks(range(3), [MN[m] for m in MC]); ax.set_ylim(0.6, 0.86); ax.set_title(f"{t.capitalize()} detection", loc="left"); ax.grid(axis="y", color=GRID, zorder=0)
axs[0].set_ylabel("mAP50-95")
fig.suptitle("Accuracy: bar = mean +/- 95% CI over 5 folds, dots = individual folds", x=0.01, ha="left", fontsize=12); fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(O / "06_accuracy_bars.png", dpi=200); plt.close(fig)

# 7 latency bars -------------------------------------------------------------------------------
Lt = json.load(open(R / "latency.json"))["models"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2, 1]})
for ax, keys, ttl in [(axs[0], [("cuda_fp32", "GPU fp32"), ("cuda_fp16", "GPU fp16")], "GPU (RTX 4050 Laptop)"), (axs[1], [("cpu_fp32", "CPU fp32")], "CPU (8 threads)")]:
    w = 0.26
    for j, m in enumerate(MC):
        for i, (k, _) in enumerate(keys):
            e = Lt[m][k]["end_to_end_ms"]; x = i + (j - 1) * w
            ax.bar(x, e["mean"], width=w * 0.88, color=MC[m], yerr=e["std"], capsize=2, error_kw=dict(ecolor=INK2, lw=.8), label=MN[m] if i == 0 else None, zorder=2)
            ax.text(x, e["mean"] + e["std"] + ax.get_ylim()[1] * 0.01, f"{e['mean']:.1f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(keys)), [n for _, n in keys]); ax.set_title(ttl, loc="left"); ax.grid(axis="y", color=GRID, zorder=0); ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
axs[0].set_ylabel("End-to-end latency (ms, lower is better)"); axs[0].legend(frameon=False, loc="upper left")
fig.suptitle("Latency, batch 1, 512x512 (mean +/- std; pre + inference + post)", x=0.01, ha="left", fontsize=12); fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(O / "07_latency_bars.png", dpi=200); plt.close(fig)

# 8 accuracy vs latency (no unpaired CI bars: fold difficulty dominates them; see paired differences table) -----------
fig, ax = plt.subplots(figsize=(6.8, 4.6))
OFF = {"yolo26n": (-10, 0, "right"), "yolo26s": (10, 0, "left"), "rtdetr-l": (-12, 0, "right")}
for m in MC:
    x = Lt[m]["cuda_fp16"]["end_to_end_ms"]["mean"]; ys = []
    for t, mk in [("person", "o"), ("gender", "s")]:
        y = sm[(sm.task == t) & (sm.model == m)].map50_95.iloc[0]; ys.append(y)
        ax.scatter(x, y, marker=mk, s=70, color=MC[m], ec=BG, lw=1.5, zorder=3)
    dx, dy, ha = OFF[m]; ax.annotate(MN[m], (x, np.mean(ys)), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=11, color=INK)
ax.set_xlabel("End-to-end latency, GPU fp16 (ms, lower is better)"); ax.set_ylabel("mAP50-95 (higher is better)"); ax.grid(color=GRID, zorder=0)
ax.set_xlim(0, 16); ax.set_ylim(0.74, 0.80)
ax.scatter([], [], marker="o", color=INK2, label="person"); ax.scatter([], [], marker="s", color=INK2, label="gender"); ax.legend(frameon=False, loc="lower right")
ax.set_title("Accuracy vs. latency (means over 5 folds)", loc="left")
fig.tight_layout(); fig.savefig(O / "08_accuracy_vs_latency.png", dpi=200); plt.close(fig)
print(sorted(p.name for p in O.iterdir()))
