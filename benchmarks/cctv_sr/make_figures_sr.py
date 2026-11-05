"""Figures for the CCTV-SR deck -> results/cctv_sr/fig_sr/*.png (200 dpi).

Run `aggregate_sr.py` first (reads results/cctv_sr/sr_summary.csv) and keep the datasets in data/.
Colours: blue = YOLO26n, orange = YOLO26s, aqua = RT-DETR-L in every model chart; in the example frames and
the class-balance chart orange/blue mean Female/Male and gray means window/door.

    sr_01/02  annotated example frames (indoor, outdoor)   [contain students: NOT committed to git]
    sr_03     class balance (log scale)
    sr_04/05  mAP50-95 per task: bar = mean +/- 95% CI over folds, dots = individual folds
    sr_06     outdoor: people per held-out block and person mAP50-95 by fold
    sr_07     latency bars (from results/cctv_sr/latency.json)

Usage: python benchmarks/cctv_sr/make_figures_sr.py
"""

import glob
import json

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    DATA,
    DATASETS,
    IMG_SIZE,
    MODEL_LABELS,
    MODELS,
    RESULTS,
    STATIC_PREFIXES,
    class_names,
    read_label_file,
)
from matplotlib.patches import Rectangle
from PIL import Image

OUT = RESULTS / "fig_sr"

# --- style ------------------------------------------------------------------------------------------------
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8e8e93"
MODEL_COLORS = dict(zip(MODELS, [BLUE, ORANGE, AQUA], strict=True))
plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def class_colour(class_name: str) -> str:
    """Orange for Female_*, blue for Male_*, gray for windows/doors."""
    return ORANGE if class_name.startswith("Female") else BLUE if class_name.startswith("Male") else GRAY


def save(fig, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


# --- 01/02: annotated example frames ----------------------------------------------------------------------
def plot_example_frame(dataset: str, min_people: int, title: str, filename: str) -> None:
    """Draw the first (every 3rd) training frame that has at least `min_people` people boxes."""
    names = class_names(DATA / dataset)
    frames = sorted((DATA / dataset / "train" / "images").glob("*.jpg"))
    for image in frames[::3]:
        label_file = DATA / dataset / "train" / "labels" / (image.stem + ".txt")
        boxes = [(names[c], box) for c, box in read_label_file(label_file)]
        if sum(not n.startswith(STATIC_PREFIXES) for n, _ in boxes) >= min_people:
            break

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(Image.open(image).convert("RGB"))
    ax.axis("off")
    for name, (cx, cy, w, h) in boxes:
        ax.add_patch(
            Rectangle(
                ((cx - w / 2) * IMG_SIZE, (cy - h / 2) * IMG_SIZE),
                w * IMG_SIZE,
                h * IMG_SIZE,
                fill=False,
                ec=class_colour(name),
                lw=1.6,
                ls="--" if class_colour(name) == GRAY else "-",  # dashed = static scenery
            )
        )
    ax.set_title(title, loc="left", fontsize=13)
    legend = [("Female", ORANGE, 0.0), ("Male", BLUE, 0.17), ("Window / door (static)", GRAY, 0.30)]
    for text, colour, dx in legend:
        fig.text(0.04 + dx, 0.03, "■", color=colour, fontsize=13)
        fig.text(0.07 + dx, 0.032, text, color=INK2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, filename)


# --- 03: class balance --------------------------------------------------------------------------------------
def plot_class_counts() -> None:
    """Boxes per class over all frames of each dataset, log scale (windows dwarf everything else)."""
    counts = {}
    for dataset in DATASETS:
        names = class_names(DATA / dataset)
        per_class = dict.fromkeys(names, 0)
        for path in (DATA / dataset).glob("*/labels/*.txt"):
            for class_id, _ in read_label_file(path):
                per_class[names[class_id]] += 1
        counts[dataset] = per_class

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [3, 9]})
    for ax, dataset in zip(axs, DATASETS, strict=True):
        per_class = counts[dataset]
        keys = list(per_class)[::-1]
        ax.barh(keys, [per_class[k] for k in keys], color=[class_colour(k) for k in keys], height=0.62, zorder=2)
        ax.set_xscale("log")
        ax.grid(axis="x", color=GRID, zorder=0)
        for k in keys:
            ax.text(per_class[k] * 1.12, k, f"{per_class[k]:,}", va="center", fontsize=9.5, color=INK2)
        ax.set_xlim(1, max(per_class.values()) * 8)
        ax.set_title(f"{dataset.capitalize()} ({sum(per_class.values()):,} boxes)", loc="left")
        ax.set_xlabel("boxes (log scale)")
    fig.suptitle("Class balance: windows/doors (gray) dominate, especially outdoors", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "sr_03_class_counts.png")


# --- 04/05: accuracy per task ---------------------------------------------------------------------------------
def plot_model_results(
    runs: pd.DataFrame, summary: pd.DataFrame, dataset: str, tasks: list[str], filename: str, title: str
) -> None:
    """One panel per task: bars = mean mAP50-95 +/- 95% CI, white dots = the individual folds."""
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks) + 0.6, 4.4), sharey=True)
    for ax, task in zip(axs, tasks, strict=True):
        rows = summary[(summary.dataset == dataset) & (summary.task == task)]
        n_folds = int(rows.folds_used.iloc[0])
        folds = [int(x) for x in rows.folds.iloc[0].split(",")]  # folds shared by all three models
        for i, model in enumerate(MODELS):
            r = rows[rows.model == model].iloc[0]
            per_fold = (
                runs[(runs.dataset == dataset) & (runs.task == task) & (runs.model == model) & runs.fold.isin(folds)]
                .sort_values("fold")
                .map50_95
            )
            ax.bar(
                i, r.map50_95, width=0.6, color=MODEL_COLORS[model], yerr=r.map50_95_ci, capsize=4,
                error_kw={"ecolor": INK2, "lw": 1}, zorder=2,
            )  # fmt: skip
            ax.scatter(
                np.full(len(per_fold), i) + np.linspace(-0.14, 0.14, len(per_fold)),
                per_fold, s=14, color="white", ec=INK2, lw=0.8, zorder=3,
            )  # fmt: skip
            ax.text(i, 0.02, f"{r.map50_95:.2f}", ha="center", color="white", fontsize=11, fontweight="bold", zorder=4)
        ax.set_xticks(range(3), [MODEL_LABELS[m] for m in MODELS], rotation=15)
        ax.set_title(f"{task} ({n_folds} folds)", loc="left", fontsize=11.5)
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.set_ylim(0, 0.9)
    axs[0].set_ylabel("mAP50-95")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, filename)


# --- 06: why outdoor folds disagree ----------------------------------------------------------------------------
def plot_outdoor_fold_density(runs: pd.DataFrame) -> None:
    """People boxes in each outdoor test block (left) next to person mAP50-95 per fold (right)."""
    people = []
    for k in range(5):
        label_dir = DATA / "sr/outdoor/person" / f"fold{k}/test/labels"
        people.append(sum(len(path.read_text().splitlines()) for path in label_dir.glob("*.txt")))
    by_fold = runs[(runs.dataset == "outdoor") & (runs.task == "person")].groupby(["fold", "model"]).map50_95.mean().unstack()

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
    axs[0].bar([f"Fold {k}" for k in range(5)], people, color=GRAY, width=0.6, zorder=2)
    axs[0].grid(axis="y", color=GRID, zorder=0)
    for i, n in enumerate(people):
        axs[0].text(i, n + 40, f"{n:,}", ha="center", fontsize=10, color=INK2)
    axs[0].set_title("People boxes in each held-out block (~187 frames)", loc="left", fontsize=11)
    axs[0].set_ylabel("people boxes")

    for model in MODELS:
        axs[1].plot(
            range(5),
            by_fold[model].reindex(range(5)),
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            lw=1.6,
        )
    axs[1].set_xticks(range(5), [f"Fold {k}" for k in range(5)])
    axs[1].set_ylabel("person mAP50-95")
    axs[1].grid(axis="y", color=GRID)
    axs[1].legend(frameon=False, fontsize=9)
    axs[1].set_title("Person mAP50-95 by fold", loc="left", fontsize=11)
    fig.suptitle(
        "Outdoor folds are very different: people are sparse early and crowded late",
        x=0.01,
        ha="left",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "sr_06_outdoor_fold_density.png")


# --- 07: latency -----------------------------------------------------------------------------------------------
def plot_latency() -> None:
    """End-to-end latency (mean +/- std) per model on GPU fp32/fp16 and CPU, from results/cctv_sr/latency.json."""
    latency = json.load(open(RESULTS / "latency.json"))["models"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    panels = [
        (axs[0], [("cuda_fp32", "GPU fp32"), ("cuda_fp16", "GPU fp16")], "GPU (RTX 4050 Laptop)"),
        (axs[1], [("cpu_fp32", "CPU fp32")], "CPU (8 threads)"),
    ]
    width = 0.26
    for ax, settings, title in panels:
        for j, model in enumerate(MODELS):
            for i, (key, _) in enumerate(settings):
                ms = latency[model][key]["end_to_end_ms"]
                x = i + (j - 1) * width  # three bars per group, centred
                ax.bar(
                    x, ms["mean"], width=width * 0.88, color=MODEL_COLORS[model], yerr=ms["std"], capsize=2,
                    error_kw={"ecolor": INK2, "lw": 0.8}, label=MODEL_LABELS[model] if i == 0 else None, zorder=2,
                )  # fmt: skip
                ax.text(
                    x,
                    ms["mean"] + ms["std"] + ax.get_ylim()[1] * 0.01,
                    f"{ms['mean']:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=INK2,
                )
        ax.set_xticks(range(len(settings)), [label for _, label in settings])
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    axs[0].set_ylabel("End-to-end latency (ms, lower is better)")
    axs[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Latency, batch 1, 512x512 (mean +/- std; pre + inference + post)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "sr_07_latency_bars.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(RESULTS / "sr/*.json"))])
    summary = pd.read_csv(RESULTS / "sr_summary.csv")

    plot_example_frame("indoor", 14, "Indoor: classroom, one fixed camera", "sr_01_indoor_example.png")
    plot_example_frame("outdoor", 4, "Outdoor: corridor, one fixed camera", "sr_02_outdoor_example.png")
    plot_class_counts()
    suffix = "bar = mean +/- 95% CI over folds, dots = folds (seed 0)"
    plot_model_results(
        runs,
        summary,
        "indoor",
        ["person", "gender", "allclass"],
        "sr_04_indoor_results.png",
        f"Indoor: mAP50-95, {suffix}",
    )
    plot_model_results(
        runs,
        summary,
        "outdoor",
        ["person", "gender", "posture", "allclass"],
        "sr_05_outdoor_results.png",
        f"Outdoor: mAP50-95, {suffix}",
    )
    plot_outdoor_fold_density(runs)
    plot_latency()


if __name__ == "__main__":
    main()
