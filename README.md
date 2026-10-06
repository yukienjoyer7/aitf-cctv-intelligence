# AITF CCTV Intelligence

Benchmark of **YOLO26n, YOLO26s and RT-DETR-L** for person, gender and posture detection on classroom / corridor CCTV, reporting accuracy (mAP50, mAP50-95) and latency.

Status: **weekly progress, not final.** The main benchmark (CCTV-SR indoor/outdoor) is one seed, and some outdoor folds are missing. See `results/weekly_progress_ppt.md`.

## Method in brief
- **Leakage-aware splits.** The provided Roboflow splits are random over consecutive frames of one fixed camera, so they leak near-duplicates. All frames are pooled, sorted by frame ID and cut into 5 contiguous time blocks, with a buffer of frames dropped around each held-out block.
- **Tasks** are class mappings over the same boxes: `person`, `gender`, `posture` (outdoor only), `allclass`.
- Final-epoch weights are evaluated, with no checkpoint selection or threshold tuning on test data. Only threshold-free AP is taken from Ultralytics.
- Comparisons between models are paired per fold.

## Layout
| Path | Contents |
|---|---|
| `src/prep_sr.py`, `sr_eval.py`, `run_sr.sh`, `aggregate_sr.py` | **CCTV-SR benchmark** (the main experiment) |
| `src/profile_ds.py`, `profile_ds2.py` | Dataset profiling (classes, box sizes, leakage, static objects) |
| `src/prep_data.py`, `train_eval.py`, `run_all.sh`, `aggregate.py`, `latency.py` | Earlier AITF classroom pilot and the latency benchmark |
| `src/prep_ir.py`, `ir_eval.py`, `run_ir.sh`, `aggregate_ir.py` | Simulated-infrared pilot (YOLO26n) |
| `src/fisheye_eval.py` | Fisheye dataset pilot (YOLO26n) |
| `src/make_figures*.py` | Figures |
| `results/` | Per-run JSON, summary CSVs, figures, slide content |

## Reproduce
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# put the Roboflow YOLO exports in data/indoor and data/outdoor (unzipped), then:
.venv/bin/python src/prep_sr.py
src/run_sr.sh                      # long: tens of GPU-hours for all seeds
.venv/bin/python src/aggregate_sr.py
```
Pretrained weights (`yolo26n.pt`, `yolo26s.pt`, `rtdetr-l.pt`) are downloaded by Ultralytics on first use.

## Not included
Datasets (CC BY 4.0 Roboflow exports of CCTV footage of students), model weights, logs, and the figures that show raw frames. Re-create the data folders with the `prep_*` scripts.
