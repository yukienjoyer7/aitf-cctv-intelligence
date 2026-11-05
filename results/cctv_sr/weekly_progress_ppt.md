# Weekly progress deck: CCTV-SR benchmark (content for slides)

> **STYLE INSTRUCTION FOR THE SLIDE BUILDER: follow the PPT template that was provided for the visual style (layout, fonts, colours, title/footer treatment).**
> This file supplies **content only**: slide titles, bullet text, tables, figure files and speaker notes. Do not invent a new visual style. The PNG figures in `fig_sr/` use a neutral light background and can be placed on the template as images. The tables below should be re-drawn in the template's table style.

**Status of this deck: weekly progress, not final.** The benchmark was stopped early on purpose so the deck could be presented on time. Single seed, and some outdoor folds are missing. Every slide that depends on this says so.

Figures: `results/fig_sr/` (PNG, 200 dpi). Numbers: `results/sr_summary.csv`, `results/sr_paired.csv`, `results/sr/*.json`, `results/latency.json`.

---

## Slide 1: Title
**Benchmarking YOLO26 and RT-DETR on CCTV-SR: person, gender and posture detection**
Weekly progress update. Subtitle: indoor classroom + corridor CCTV, accuracy and latency.

*Notes:* This week the project moved to the new CCTV-SR datasets (indoor and outdoor). Earlier work on the older classroom dataset was a pilot and is not reported here.

---

## Slide 2: What we did this week
- Received and analysed the two new datasets: **CCTV-SR indoor** and **CCTV-SR outdoor**.
- Audited them for label and split problems, and rebuilt the splits so they do not leak.
- Benchmarked **YOLO26n, YOLO26s and RT-DETR-L** on both datasets, across several tasks.
- Stopped the run early so this deck could be presented: **results are preliminary**.

**Coverage so far**

| | Planned | Completed |
|---|---|---|
| Indoor: person, gender, allclass | 5 folds x 3 seeds | 5 folds x **1 seed** |
| Outdoor: person | 5 folds x 1 seed | 5 folds x 1 seed |
| Outdoor: gender, posture, allclass | 5 folds x 1 seed | **4 of 5 folds** x 1 seed (fold 4 missing) |

96 training runs finished in total (45 indoor, 51 outdoor).

*Notes:* Fold 4 of the outdoor dataset is the busiest block (about 2,800 people boxes), so its absence in three of the tasks matters; see slide 9.

---

## Slide 3: The two datasets
| | **Indoor** | **Outdoor** |
|---|---|---|
| Scene | Classroom, one fixed camera ("Kelas X4") | Second-floor corridor, one fixed camera ("Lantai 2 Cam 4") |
| Images | 125 | 934 |
| Boxes | 2,575 | 36,448 |
| Classes | 3: Female_Sitting, Male_Sitting, Window_Close | 9: Door_Close, Window_Close, Window_Open, and Female / Male x Sitting / Standing / Walking |
| People | 2,009 boxes (79% Female, 21% Male), about 16 per image | 4,517 boxes (25% Female, 75% Male), about 5 per image |
| Image size | 512 x 512 (stretched) | 512 x 512 (stretched) |

Figures (place side by side): `fig_sr/sr_01_indoor_example.png`, `fig_sr/sr_02_outdoor_example.png`
(orange = Female, blue = Male, gray dashed = window/door).

*Notes:* "Outdoor" is really an open-sided corridor. Both datasets are a single fixed camera, so every frame shows the same scene. Labels were provided as boxes plus a few hundred 4-point polygons (124 indoor, 933 outdoor), which we converted to enclosing boxes.

---

## Slide 4: Class balance
Figure: `fig_sr/sr_03_class_counts.png` (log scale)

- **Windows and doors are 88% of all outdoor boxes** (31,931 of 36,448) and 22% of indoor boxes.
- Outdoor people classes are small and imbalanced: Female_Sitting has **7** boxes in total, Male_Sitting 142.
- Gender is skewed in opposite directions: indoor 79% Female, outdoor 75% Male.

*Notes:* Because windows never move, a model can score highly on them by memorising positions. That is why we report person, gender and posture as the main results, and the all-class run separately.

---

## Slide 5: Data quality findings
1. **The provided train/valid/test splits leak.** They are random over consecutive frames of one camera. For every indoor valid/test frame, a train frame exists within 5 frame IDs (median distance 1); outdoor 88%. Pixel difference to the nearest train frame is about 0.6 (0-255 scale), versus 2.8 (indoor) and 10.7 (outdoor) between random train pairs.
2. **Static objects leak by position.** 98-100% of valid/test window and door boxes have a same-class train box at almost the same pixel position (IoU >= 0.7).
3. **Outdoor has duplicate people boxes with conflicting labels.** 1,873 pairs of people boxes overlap at IoU >= 0.8, in 501 of 934 frames. In **723 pairs the two boxes have different gender labels**, and in 133 different posture labels. Indoor has none. This puts a ceiling on outdoor gender and posture accuracy.
4. **Outdoor people are small and few.** About 5 people per image, 40-45% of images contain no person, and standing people are roughly 26-30 px tall on the 512 image.
5. **The camera sees two stretches of time outdoors.** Frame IDs jump from 90 to 306.

*Notes:* The leakage is the main reason we did not use the provided splits. The duplicate-label issue is a data problem to fix with the annotators, not a model problem. We did not clean it for this run.

---

## Slide 6: Method
**Splits (leakage-aware).** All provided splits were pooled, sorted by frame ID and cut into **5 contiguous time blocks**. Each block is held out once; train frames within 5 (indoor) or 10 (outdoor) frame IDs of the block are dropped.
- Indoor: 25 test frames per fold, 90-95 train frames.
- Outdoor: about 187 test frames per fold, 727-738 train frames.
- Frames with no relevant box stay in as background images.

**Tasks** (same images, different class mapping):

| Task | Classes | Indoor | Outdoor |
|---|---|---|---|
| Person | 1: any person | yes | yes |
| Gender | 2: Female, Male | yes | yes |
| Posture | 3: Sitting, Standing, Walking | no (sitting only) | yes |
| All-class | every original class incl. windows/doors | 3 classes | 9 classes |

**Models.** YOLO26n, YOLO26s, RT-DETR-L, all COCO-pretrained, fine-tuned. 512 px, AMP. Indoor 100 epochs, outdoor 30 epochs (outdoor has about 8x more training frames, so more steps per epoch). Batch 8 (RT-DETR-L batch 4 to fit the 6 GB GPU). Final-epoch weights, no checkpoint selection.

**Metrics.** mAP50 and mAP50-95 (threshold-free). For gender and posture also **matched class accuracy**: predictions with confidence >= 0.25 (fixed in advance) matched to ground truth at IoU >= 0.5, then checked for the right class. Mean +/- 95% confidence interval across folds, and **paired per-fold comparisons** between models.

*Notes:* Precision, recall and F1 are deliberately left out, because the usual implementation picks its threshold on the evaluation data. All numbers are from one seed.

---

## Slide 7: Results, indoor
Figure: `fig_sr/sr_04_indoor_results.png`

mAP50-95, mean +/- 95% CI over 5 folds (seed 0):

| Task | YOLO26n | YOLO26s | RT-DETR-L |
|---|---|---|---|
| Person | 0.686 +/- 0.052 | 0.727 +/- 0.044 | 0.727 +/- 0.037 |
| Gender | 0.666 +/- 0.080 | 0.717 +/- 0.058 | 0.717 +/- 0.054 |
| All-class (incl. windows) | 0.656 +/- 0.056 | 0.694 +/- 0.051 | 0.692 +/- 0.045 |

mAP50: person 0.961 / 0.978 / 0.982; gender 0.933 / 0.960 / 0.977; all-class 0.918 / 0.935 / 0.944 (n / s / RT-DETR-L).
Matched gender accuracy: **0.999 / 0.999 / 1.000**.

Paired per-fold differences in mAP50-95 (all three tasks agree):
- YOLO26s - YOLO26n: **+0.038 to +0.051**, better in 5 of 5 folds, interval excludes zero.
- RT-DETR-L - YOLO26n: **+0.036 to +0.051**, better in 5 of 5 folds.
- RT-DETR-L - YOLO26s: **about 0** (-0.002 to 0.000), no real difference.

*Notes:* YOLO26s matches RT-DETR-L on accuracy here. Gender accuracy of about 100% is a ceiling effect: it is a single classroom, where the labels likely follow appearance such as hijab. It is not evidence of a hard gender problem being solved.

---

## Slide 8: Results, outdoor
Figure: `fig_sr/sr_05_outdoor_results.png`

mAP50-95, mean +/- 95% CI over folds (seed 0). Person uses 5 folds, the others **4 folds (0-3)**:

| Task | YOLO26n | YOLO26s | RT-DETR-L |
|---|---|---|---|
| Person (5 folds) | 0.349 +/- 0.135 | 0.379 +/- 0.137 | 0.320 +/- 0.111 |
| Gender (4 folds) | 0.294 +/- 0.311 | 0.330 +/- 0.368 | 0.335 +/- 0.366 |
| Posture (4 folds) | 0.216 +/- 0.065 | 0.256 +/- 0.067 | 0.239 +/- 0.126 |
| All-class (4 folds) | 0.244 +/- 0.089 | 0.290 +/- 0.101 | 0.283 +/- 0.099 |

Matched class accuracy: gender 0.787 / 0.794 / 0.864; posture 0.808 / 0.804 / 0.824 (n / s / RT-DETR-L; wide intervals, see notes).

Paired comparisons:
- **Person:** YOLO26s beats YOLO26n by +0.029 +/- 0.017 (5 of 5 folds). **RT-DETR-L is worse than YOLO26s by 0.058 +/- 0.032** (0 of 5 folds better) and slightly worse than YOLO26n (-0.029 +/- 0.027).
- **Gender, posture, all-class:** no reliable difference between models (intervals include zero, only 4 folds).

*Notes:* Outdoor is far harder than indoor (mAP50-95 about 0.3 vs about 0.7): people are small and sparse, and the labels contain conflicting duplicates (slide 5). The confidence intervals are very wide, especially for gender, because a few folds have very few gender examples. Do not rank models on gender or posture from this data.

---

## Slide 9: Why outdoor folds disagree
Figure: `fig_sr/sr_06_outdoor_fold_density.png`

People boxes in each held-out block: 375, 172, 130, **1,038**, **2,802** (folds 0 to 4).

Outdoor person mAP50-95 by fold (seed 0):

| Fold | YOLO26n | YOLO26s | RT-DETR-L |
|---|---|---|---|
| 0 | 0.387 | 0.426 | 0.361 |
| 1 | 0.167 | 0.190 | 0.175 |
| 2 | 0.461 | 0.477 | 0.415 |
| 3 | 0.366 | 0.415 | 0.327 |
| 4 | 0.365 | 0.384 | 0.322 |

- The recording is quiet early (130-375 people boxes per block) and crowded late (1,038 and 2,802). The folds are very different regimes.
- Fold 1 is hardest for every model (about 0.17-0.19).
- The model ranking is similar in every fold: **YOLO26s on top, RT-DETR-L lowest in 4 of 5 folds**.

*Notes:* The three outdoor tasks without fold 4 are missing the busiest block, so their means are likely optimistic or at least not comparable with person's 5-fold mean.

---

## Slide 10: Latency and model size
Figure: `fig_sr/sr_07_latency_bars.png`

Batch 1, 512 x 512, RTX 4050 Laptop GPU, end-to-end (preprocess + model + postprocess), mean ms with 95th percentile:

| Model | Params | GFLOPs | GPU fp32 | GPU fp16 | CPU fp32 (8 threads) | Training time per run |
|---|---|---|---|---|---|---|
| YOLO26n | 2.5 M | 3.7 | 4.6 ms (p95 5.0) | 4.9 ms (5.6) | 25 ms (27) | indoor 1.4 min, outdoor 2.1 min |
| YOLO26s | 9.9 M | 14.5 | 5.5 ms (5.8) | 5.1 ms (5.6) | 67 ms (75) | indoor 2.3 min, outdoor 3.4 min |
| RT-DETR-L | 32.8 M | 72.1 | 19.3 ms (20.0) | 13.4 ms (14.0) | 322 ms (336) | indoor 9.4 min, outdoor 16 min |

- RT-DETR-L is about **3.5-4.2x slower than the YOLO26 models on GPU fp32 (2.6-2.7x on fp16)** and about **13x slower than YOLO26n on CPU**.
- YOLO26n and YOLO26s are close on GPU because tiny models are limited by kernel-launch overhead; the CPU numbers show the real compute gap (2.7x).

*Notes:* The latency benchmark was measured earlier with models trained on the previous pilot dataset. Latency depends on the architecture, not the data, so it applies here; the number of output classes changes it only marginally. It was not re-measured for the 3- and 9-class heads. Laptop GPU, so thermal throttling is possible.

---

## Slide 11: Takeaways
1. **YOLO26s is the best accuracy/cost choice so far.** Indoor it ties RT-DETR-L (paired difference about 0) at about 2.6-3.5x lower GPU latency (fp16 / fp32) and about 4x less training time. Outdoors it is the top model for person detection.
2. **YOLO26n is the lightest option** (25 ms on CPU) and loses about 0.04 mAP50-95 versus YOLO26s indoor (about 0.03 outdoor person).
3. **RT-DETR-L is not worth its cost here.** Heavier and slower, with no accuracy gain over YOLO26s, and worse on outdoor person detection.
4. **Outdoor is the hard case.** About 0.3 mAP50-95 versus 0.7 indoor, driven by tiny, sparse people, a skewed fold structure and conflicting duplicate labels.
5. **Gender results are not trustworthy yet.** Indoor accuracy of about 100% is a ceiling effect from one classroom; outdoor accuracy of about 79-86% is capped by 723 conflicting duplicate-label pairs.

---

## Slide 12: Limitations
- **One seed**, so no seed-to-seed variance. Seeds 1-2 were planned and stopped.
- **Outdoor gender, posture and all-class use 4 of 5 folds**; the busiest block (fold 4) is missing for them.
- **Single fixed camera per dataset**: results measure generalisation across time in that scene, not to new rooms or cameras.
- **Outdoor label noise** (duplicate boxes with conflicting gender/posture) was not cleaned.
- **Very rare classes** (outdoor Female_Sitting has 7 boxes in total) cannot be evaluated reliably; they are skipped in folds where they are absent.
- **Different epoch budgets**: indoor 100 epochs, outdoor 30; RT-DETR-L used batch 4 and the others batch 8.
- **Gender labels may track attire** (for example hijab), which was not verified on these datasets.
- Latency numbers are from a laptop GPU and from models of the previous dataset.

---

## Slide 13: Next steps
1. **Finish the outdoor run**: fold 4 for gender, posture and all-class, then extra seeds for error bars.
2. **Clean the outdoor labels**: resolve duplicate boxes with conflicting gender and posture.
3. **Add a crop-based gender / posture classifier** (accuracy, F1, confusion matrix) as a cleaner alternative to the detector-only setup.
4. **Re-measure latency** with the final 3-class and 9-class heads, and test an exported (ONNX/TensorRT) runtime.
5. **Validate on a second camera or room** to measure generalisation beyond a single scene.

---

## Appendix A: Full summary tables (seed 0, mean +/- 95% CI over folds used)

**Indoor (5 folds)**

| Task | Model | mAP50 | mAP50-95 | Matched class acc. | Train min |
|---|---|---|---|---|---|
| Person | YOLO26n | 0.961 +/- 0.039 | 0.686 +/- 0.052 | n/a | 1.4 |
| Person | YOLO26s | 0.978 +/- 0.019 | 0.727 +/- 0.044 | n/a | 2.3 |
| Person | RT-DETR-L | 0.982 +/- 0.010 | 0.727 +/- 0.037 | n/a | 9.4 |
| Gender | YOLO26n | 0.933 +/- 0.085 | 0.666 +/- 0.080 | 0.999 | 2.0 |
| Gender | YOLO26s | 0.960 +/- 0.048 | 0.717 +/- 0.058 | 0.999 | 2.8 |
| Gender | RT-DETR-L | 0.977 +/- 0.016 | 0.717 +/- 0.054 | 1.000 | 8.8 |
| All-class | YOLO26n | 0.918 +/- 0.059 | 0.656 +/- 0.056 | n/a | 1.2 |
| All-class | YOLO26s | 0.935 +/- 0.040 | 0.694 +/- 0.051 | n/a | 2.3 |
| All-class | RT-DETR-L | 0.944 +/- 0.020 | 0.692 +/- 0.045 | n/a | 8.1 |

**Outdoor**

| Task (folds) | Model | mAP50 | mAP50-95 | Matched class acc. | Train min |
|---|---|---|---|---|---|
| Person (5) | YOLO26n | 0.526 +/- 0.188 | 0.349 +/- 0.135 | n/a | 2.1 |
| Person (5) | YOLO26s | 0.546 +/- 0.165 | 0.379 +/- 0.137 | n/a | 3.4 |
| Person (5) | RT-DETR-L | 0.417 +/- 0.116 | 0.320 +/- 0.111 | n/a | 16.2 |
| Gender (4) | YOLO26n | 0.421 +/- 0.396 | 0.294 +/- 0.311 | 0.787 | 2.0 |
| Gender (4) | YOLO26s | 0.448 +/- 0.454 | 0.330 +/- 0.368 | 0.794 | 3.3 |
| Gender (4) | RT-DETR-L | 0.436 +/- 0.439 | 0.335 +/- 0.366 | 0.864 | 15.8 |
| Posture (4) | YOLO26n | 0.341 +/- 0.077 | 0.216 +/- 0.065 | 0.808 | 2.1 |
| Posture (4) | YOLO26s | 0.370 +/- 0.079 | 0.256 +/- 0.067 | 0.804 | 3.3 |
| Posture (4) | RT-DETR-L | 0.325 +/- 0.163 | 0.239 +/- 0.126 | 0.824 | 16.1 |
| All-class (4) | YOLO26n | 0.327 +/- 0.088 | 0.244 +/- 0.089 | n/a | 2.2 |
| All-class (4) | YOLO26s | 0.377 +/- 0.110 | 0.290 +/- 0.101 | n/a | 3.5 |
| All-class (4) | RT-DETR-L | 0.358 +/- 0.092 | 0.283 +/- 0.099 | n/a | 16.3 |

## Appendix B: Paired per-fold differences in mAP50-95 (A minus B; mean +/- 95% CI; "folds A better" = folds where A > B)

| Dataset | Task | A vs B | Difference | Folds A better |
|---|---|---|---|---|
| Indoor | Person | YOLO26s vs YOLO26n | +0.041 +/- 0.012 | 5 / 5 |
| Indoor | Person | RT-DETR-L vs YOLO26n | +0.041 +/- 0.018 | 5 / 5 |
| Indoor | Person | RT-DETR-L vs YOLO26s | -0.001 +/- 0.016 | 4 / 5 |
| Indoor | Gender | YOLO26s vs YOLO26n | +0.051 +/- 0.024 | 5 / 5 |
| Indoor | Gender | RT-DETR-L vs YOLO26n | +0.051 +/- 0.028 | 5 / 5 |
| Indoor | Gender | RT-DETR-L vs YOLO26s | 0.000 +/- 0.009 | 3 / 5 |
| Indoor | All-class | YOLO26s vs YOLO26n | +0.038 +/- 0.013 | 5 / 5 |
| Indoor | All-class | RT-DETR-L vs YOLO26n | +0.036 +/- 0.022 | 5 / 5 |
| Indoor | All-class | RT-DETR-L vs YOLO26s | -0.002 +/- 0.021 | 2 / 5 |
| Outdoor | Person | YOLO26s vs YOLO26n | +0.029 +/- 0.017 | 5 / 5 |
| Outdoor | Person | RT-DETR-L vs YOLO26n | -0.029 +/- 0.027 | 1 / 5 |
| Outdoor | Person | RT-DETR-L vs YOLO26s | -0.058 +/- 0.032 | 0 / 5 |
| Outdoor | Gender | YOLO26s vs YOLO26n | +0.036 +/- 0.064 | 3 / 4 |
| Outdoor | Gender | RT-DETR-L vs YOLO26n | +0.041 +/- 0.080 | 3 / 4 |
| Outdoor | Gender | RT-DETR-L vs YOLO26s | +0.005 +/- 0.039 | 2 / 4 |
| Outdoor | Posture | YOLO26s vs YOLO26n | +0.040 +/- 0.031 | 4 / 4 |
| Outdoor | Posture | RT-DETR-L vs YOLO26n | +0.023 +/- 0.068 | 3 / 4 |
| Outdoor | Posture | RT-DETR-L vs YOLO26s | -0.018 +/- 0.085 | 1 / 4 |
| Outdoor | All-class | YOLO26s vs YOLO26n | +0.046 +/- 0.069 | 3 / 4 |
| Outdoor | All-class | RT-DETR-L vs YOLO26n | +0.038 +/- 0.062 | 4 / 4 |
| Outdoor | All-class | RT-DETR-L vs YOLO26s | -0.007 +/- 0.016 | 1 / 4 |

## Appendix C: Figure index (`results/fig_sr/`)

| File | Use on slide |
|---|---|
| `sr_01_indoor_example.png` | 3 |
| `sr_02_outdoor_example.png` | 3 |
| `sr_03_class_counts.png` | 4 |
| `sr_04_indoor_results.png` | 7 |
| `sr_05_outdoor_results.png` | 8 |
| `sr_06_outdoor_fold_density.png` | 9 |
| `sr_07_latency_bars.png` | 10 |

Colour code in the model charts: blue = YOLO26n, orange = YOLO26s, aqua = RT-DETR-L. In the example images and the class-balance chart, orange/blue mean Female/Male and gray means window/door.
