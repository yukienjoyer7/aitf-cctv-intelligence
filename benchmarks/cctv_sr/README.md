# CCTV-SR historical benchmark

This is a historical, preliminary comparison of YOLO26n, YOLO26s and RT-DETR-L on 125 indoor classroom and 934 outdoor corridor frames, each from one fixed camera. Provided random splits leak adjacent frames. The scripts pool those splits and form five contiguous frame-ID blocks, purging training frames within 5 indoor or 10 outdoor IDs of the held-out block. Generalization across time in these scenes does not establish generalization to new cameras.

The same original boxes are mapped to `person` (all people), `gender` (female/male annotation), `posture` (outdoor sitting/standing/walking) and `allclass` (original categories including windows/doors). These are benchmark-specific detection labels, not a student classifier or a reliable inference of gender identity. Outdoor near-duplicate boxes have conflicting labels; they were not cleaned.

Each job trains for fixed epochs and evaluates final-epoch weights without test-set checkpoint selection. AP is threshold-free; matched gender/posture classification accuracy uses confidence 0.25 and IoU 0.5. Indoor uses 100 epochs, outdoor 30; YOLO batch 8 and RT-DETR-L batch 4. Aggregation uses folds completed by all three models for each dataset/task and paired differences across folds, with 95% t intervals. Only seed 0 was completed; outdoor gender, posture and allclass lack fold 4. Indoor person mAP50-95 was approximately 0.686, 0.727, 0.727, outdoor 0.349, 0.379, 0.320, respectively. These are observations, not operational guarantees. See the [historical slides](../../results/cctv_sr/weekly_progress_ppt.md) for details.

## Reproduction and outputs

Install the existing requirements and place the original exports under `data/indoor` and `data/outdoor`. From the repository root:

```bash
.venv/bin/python benchmarks/cctv_sr/prep_sr.py
benchmarks/cctv_sr/run_sr.sh
.venv/bin/python benchmarks/cctv_sr/aggregate_sr.py
.venv/bin/python benchmarks/cctv_sr/make_figures_sr.py
```

Old `src/<script>` commands map to `benchmarks/cctv_sr/<script>`. Preparation replaces generated task fold directories in `data/sr`; the runner resumes by skipping existing job JSONs in `results/cctv_sr/sr`. It deletes checkpoints under its benchmark run directory after evaluation. Future model handoffs must retain and export their own checkpoints. Aggregation and figures overwrite archived output files if run in place. Source datasets, pretrained weights, and raw CCTV figures are not distributed; model libraries may download weights on first use.

Recorded summaries, paired comparisons, per-job JSONs and plots are in [results/cctv_sr](../../results/cctv_sr/). The latency JSON was measured on a laptop RTX 4050 using pilot fold-0 weights, not final reported heads. After the original experiment, scripts were refactored with intended logic unchanged; only preparation was rerun then.
