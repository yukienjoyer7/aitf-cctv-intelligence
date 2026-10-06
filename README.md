# AITF CCTV Intelligence: CCTV-SR benchmark

Benchmark of **YOLO26n, YOLO26s and RT-DETR-L** for person, gender and posture detection on two fixed CCTV cameras
(a classroom and a corridor), reporting accuracy (mAP50, mAP50-95) and latency.

> **Status: weekly progress, not final.** One seed, and outdoor gender/posture/all-class are missing fold 4.
> Slide content with all numbers: [`results/weekly_progress_ppt.md`](results/weekly_progress_ppt.md).

## Method in brief
- **Leakage-aware splits.** The provided Roboflow splits are random over consecutive frames of one fixed camera, so test
  frames have near-copies in train. All frames are pooled, sorted by frame id and cut into 5 contiguous time blocks; train
  frames within 5 (indoor) / 10 (outdoor) frame ids of a test block are dropped.
- **Tasks** are class mappings over the same boxes: `person`, `gender`, `posture` (outdoor only), `allclass`.
- **No tuning on test data.** Final-epoch weights are evaluated (no checkpoint selection) and only threshold-free AP is
  taken from Ultralytics. The class-accuracy metric uses a fixed confidence of 0.25.
- **Paired comparisons.** Models are compared fold by fold on the folds all three finished, with 95% CIs across folds.

## Pipeline
```
prep_sr.py ──> sr_eval.py (via run_sr.sh) ──> aggregate_sr.py ──> make_figures_sr.py
 folds          one training job + eval         summary tables        slide figures
```

| Script (`src/`) | What it does |
|---|---|
| `common.py` | Shared paths, constants, label parsing (polygon -> box) and IoU |
| `prep_sr.py` | Pool the provided splits and build the blocked folds for every task |
| `sr_eval.py` | Train one (dataset, task, model, fold, seed) job and evaluate the final-epoch weights |
| `run_sr.sh` | Run all jobs; resumable because finished jobs are skipped |
| `aggregate_sr.py` | Mean +/- 95% CI over shared folds and paired differences -> `results/sr_summary.csv`, `sr_paired.csv` |
| `make_figures_sr.py` | Slide figures -> `results/fig_sr/` |
| `latency.py` | Batch-1 latency, FPS, memory, params and GFLOPs -> `results/latency.json` |
| `profile_dataset.py` | Dataset sizes, class balance and box sizes |
| `profile_leakage.py` | Frame and static-label leakage, people per image |
| `check_label_conflicts.py` | Overlapping people boxes with conflicting gender/posture labels |

Results: `results/sr/*.json` (one file per job), `results/sr_summary.csv`, `results/sr_paired.csv`, `results/latency.json`.

## Reproduce
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# unzip the Roboflow YOLO exports into data/indoor and data/outdoor, then:
.venv/bin/python src/prep_sr.py
src/run_sr.sh                        # long GPU job; stop and restart freely
.venv/bin/python src/aggregate_sr.py
.venv/bin/python src/make_figures_sr.py
```
Pretrained weights (`yolo26n.pt`, `yolo26s.pt`, `rtdetr-l.pt`) are downloaded by Ultralytics on first use. Code style:
`ruff format src && ruff check src` (config in `pyproject.toml`).

## Notes and limitations
- Single fixed camera per dataset: results measure generalisation across time in that scene, not to new rooms.
- The outdoor labels contain many near-identical people boxes with conflicting gender labels (`check_label_conflicts.py`),
  which caps gender/posture accuracy; they were not cleaned.
- Indoor uses 100 epochs, outdoor 30 (about 8x more frames per fold); RT-DETR-L uses batch 4 (GPU memory), the YOLO models 8.
- `results/latency.json` was measured by an earlier revision of `latency.py` on a laptop GPU; see the note in that script.
- The scripts were refactored after the benchmark ran (comments, structure, shared helpers) with the logic left unchanged;
  only `prep_sr.py` was re-executed afterwards (its output matches the recorded per-run test counts).

## Not included
Datasets (CC BY 4.0 Roboflow exports of CCTV footage of students), model weights, logs, and the figures that show raw
frames (`sr_01`, `sr_02`). Earlier pilot experiments (an AITF classroom dataset, a simulated-infrared test and a fisheye
dataset) were removed from the working tree; they remain in git history before commit `5c3b897`.
