# Smoking event recognition

Status: exploratory baseline trained on unreviewed source labels; not production ready. Recognize observable smoking behavior or episodes, not merely a person or cigarette-like object. Define `non_smoking` and `smoking` with explicit ambiguous/occluded handling; object cues are not event labels. Candidate inputs are frames or clips, potentially person/upper-body ROIs, but no existing detector is required. Compare cue detection and ROI classification, considering temporal confirmation when frames are ambiguous.

Use license-reviewed public real imagery/clips with hard negatives including phones, pens, eating/drinking, hand-to-face gestures, vaping if excluded, and low-resolution people. Report real held-out test precision, recall, F1 and PR-AUC; use event-level measures and false alarms per camera-hour when annotated video permits. Synthetic stress tests are supplementary.

The [Khan smoking/non-smoking image dataset](https://data.mendeley.com/datasets/7b52hhzs3r/1) is a candidate for an initial ROI baseline. The publisher lists CC BY 4.0; its search-engine-sourced images require underlying rights and source review. Labels apply to whole images, so do not assign an image label automatically to every detected person. Review each person crop and account for missed detections before creating source-disjoint splits. See the [candidate registry entry](../../../../configs/datasets/smoking_khan_2020.yaml).

After reviewing dataset permissions and extracting it locally, propose person crops for review:

```bash
.venv/bin/python src/aitf_cv/tasks/smoking/propose_crops.py \
  data/smoking/public_source/images yolo26s.pt data/smoking/crop_review_v1 --device cpu
```

Replace the source directory with the actual extracted location. The output must be a new directory under ignored `data/smoking/`. `review.csv` leaves labels blank and marks missed detections and read errors for review; record `smoking`, `non_smoking`, `ambiguous`, or `rejected` after inspecting each crop and its context. `metadata.json` records the detector checkpoint hash and an uncalibrated proposal confidence. The tool does not train, select a threshold, or validate detection recall. Small or missed people and whole-image versus person-level label mismatches require explicit review.

Audit the extracted candidate before forming splits:

```bash
.venv/bin/python src/aitf_cv/tasks/smoking/audit_images.py data/smoking/public_source/unpacked/dataset
```

The version 1 archive matched the publisher checksum. It contained 2,410 images, including 400 in an undifferentiated `testing_data` folder; the archive included no test label file. SHA-256 audit found 268 duplicate image groups, of which 241 span the supplied folders, including 15 shared between labeled training and validation folders. No exact-hash group had conflicting supplied class folders. Exact hashing does not detect resized, recompressed or otherwise similar images, so audit near-duplicates and original sources before forming independent splits. Do not use the supplied folders as evidence of independent evaluation.

A second candidate, [smoking_img_final](https://huggingface.co/datasets/ccclllwww/smoking_img_final), provides image-level `not_smoking` and `smoking` labels for both training and test. The published page has no dataset card or license, and image provenance and permitted training use remain unresolved. The candidate is **not approved for training** pending rights review. A local copy of the Hugging Face Parquet conversion and its extracted labeled images are kept under ignored `data/smoking/hf_smoking_img_final/`; no media are checked in.

An exact-hash audit of this copy found 2,629 train images and 291 test images, one duplicate hash appearing in both splits, and 54 test images also present in the earlier Khan archive. Exact matching does not establish independence of remaining images, and image labels are not automatically labels for every detected person. The complete audit can be reproduced with optional `pyarrow` installed:

```bash
.venv/bin/python src/aitf_cv/tasks/smoking/audit_hf_parquet.py \
  data/smoking/hf_smoking_img_final/parquet \
  --compare-images data/smoking/public_source/unpacked/dataset
```

[Smoking people v2](https://universe.roboflow.com/thanyalak-bf8xn/smoking-people-jbes1/dataset/2) is a candidate with published `smoking person`, `non smoking person` and `cigarette` bounding-box classes. Its publisher lists CC BY 4.0, but underlying source rights, actual annotation quality, class balance and split independence remain unverified. Roboflow export requires authentication. Once a YOLO-format archive is obtained and safely extracted under `data/smoking/`, audit it first:

```bash
.venv/bin/python src/aitf_cv/tasks/smoking/audit_yolo_boxes.py \
  data/smoking/smoking_people_v2 --compare-images data/smoking/public_source/unpacked/dataset
```

The audit validates class names and normalized boxes, reports missing and empty label files without treating them as negatives, and checks exact duplicates and cross-split overlap. Repeat with the other candidate dataset if it might be mixed into training. Exact matching cannot detect near duplicates or shared recordings. Check image and annotation provenance and inspect boxes before treating the classes as reliable.

To prepare person crops for human review after resolving invalid annotations:

```bash
.venv/bin/python src/aitf_cv/tasks/smoking/propose_annotated_crops.py \
  data/smoking/smoking_people_v2 data/smoking/smoking_people_review_v1
```

The review CSV records the supplied source label but leaves `reviewed_label` blank. It does not select or certify splits or create a trained model. The output directory must be new and remain under ignored `data/smoking/`.

The local v2 export has 3,547 images and 2,740 person-labeled boxes (868 non-smoking, 1,872 smoking). Of the images, 910 have empty annotation files and 165 have only cigarette boxes; neither category should automatically become a non-smoking label. An 18-box visual spot check showed some boxes labeled as people contained objects or only parts of a person; supplied labels and box quality therefore need review. The archive SHA-256 is `831590385f9eedb76b53d527fb29d9d2917bee794d4b7619d98a404455fdd972`.

Exact-byte hashing found no cross-split or cross-dataset matches, but a 64-bit dHash screen at Hamming distance at most 3 found 20 candidate cross-split pairs and 616 images with candidate matches in the Khan dataset. A visual check of 12 sampled cross-split pairs showed the same underlying image in each pair. These pairs are candidates, not a complete near-duplicate inventory; do not trust the supplied test split as independent. A screen against smoking_img_final found 2,466 primary images with at least one candidate match; this is not a verified match count. Audit results and the 2,740 person-crop proposals remain in ignored `data/smoking/`; all crop labels await review. The source is an image dataset and includes non-CCTV imagery; it cannot establish field performance. Reproduce the candidate screen with:

```bash
.venv/bin/python src/aitf_cv/tasks/smoking/audit_near_duplicates.py \
  data/smoking/smoking_people_v2 --compare-images data/smoking/public_source/unpacked/dataset
```

## Provisional grouped split

Prepare seed-42 crop manifests with the local review proposals and source images:

```bash
PYTHONPATH=src .venv/bin/python -m aitf_cv.tasks.prepare_splits smoking data/splits/smoking_v1
```

The command requires a **new** output directory and creates `train.csv`, `val.csv`, `test.csv`, `excluded.csv`, and `metadata.json`. The manifests reference crops without copying media. It groups all boxes from an image, normalized source filenames, and transitive candidate near-duplicates using a 64-bit dHash distance at most 3 before assigning groups. Because dHash can both miss and incorrectly group images, and because source camera and subject identities are unavailable, this is not a verified source-independent split. Groups with conflicting source labels are excluded conservatively. Empty or cigarette-only annotations are not converted into non-smoking labels.

Every label in the current manifests is marked `source_unreviewed`, including test labels. Inspect and correct the person crops and obtain relevant rights before training or treating a test result as valid. Never train on overlapping images from the other local smoking datasets and then evaluate on this provisional test set. The original source splits and review proposals remain unchanged.

The selected initial crop-classifier candidate is **MobileNetV4 Conv-Small**, initialized from ImageNet-pretrained weights when available and fine-tuned for the two behavior classes. A person detector such as the local YOLO26s checkpoint can propose crops; the classifier requires a separate checkpoint. The `timm` implementation and pretrained weights are optional training dependencies. A local exploratory checkpoint and validation-selected threshold exist, but source labels remain unreviewed and neither the model nor its threshold is validated for deployment. Model choice and crop geometry remain subject to further evaluation and target-device measurements.

The exploratory training entrypoint uses the provisional manifest with explicit acknowledgement of unreviewed labels:

```bash
PYTHONPATH=src .venv/bin/python -m aitf_cv.tasks.train_baseline smoking artifacts/smoking/baseline_v1 --allow-unreviewed
```

It saves a validation-selected checkpoint and preliminary test metrics. Outputs and source media remain ignored by Git. Install the optional training dependencies separately from the historical benchmark requirements.
