"""Train ONE job (dataset, task, model, fold, seed) and evaluate it on that fold's held-out block.

Design choices that keep the evaluation leakage-free:
  * No validation set and no checkpoint selection: training runs for a fixed number of epochs and the
    FINAL-epoch weights (`last.pt`) are evaluated, so nothing is tuned on the test block.
  * mAP50 / mAP50-95 are threshold-free, so no confidence threshold is chosen on test data.
  * The only fixed threshold is conf=0.25 for the class-accuracy metric, chosen in advance.

Output: results/sr/<dataset>_<task>_<model>_f<fold>_s<seed>.json  (skipped if it already exists, so the
orchestrator `run_sr.sh` can be stopped and restarted). Checkpoints are deleted after evaluation to save disk.

Usage: python src/sr_eval.py --dataset indoor --task gender --model yolo26s --fold 0 --seed 0 --epochs 100
"""

import argparse
import json
import shutil
import time

import torch
import yaml
from torchvision.ops import box_iou
from ultralytics import RTDETR, YOLO

from common import DATA, RESULTS, ROOT

CLASS_ACC_CONF = 0.25  # fixed confidence for the class-accuracy metric
MATCH_IOU = 0.5  # a prediction "localises" a person if it overlaps a ground-truth box at IoU >= 0.5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=["indoor", "outdoor"])
    ap.add_argument("--task", required=True, choices=["person", "gender", "posture", "allclass"])
    ap.add_argument("--model", required=True, choices=["yolo26n", "yolo26s", "rtdetr-l"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    return ap.parse_args()


def count_test_instances(label_dir, n_classes: int) -> list[int]:
    """Number of ground-truth boxes per class in the held-out block (needed to judge rare classes)."""
    counts = [0] * n_classes
    for path in label_dir.glob("*.txt"):
        for line in path.read_text().splitlines():
            counts[int(line.split()[0])] += 1
    return counts


def matched_class_accuracy(model, data_yaml, imgsz: int) -> tuple[float, int]:
    """Among people the model localises, how often does it also get the class (gender/posture) right?

    Each prediction (highest confidence first) is matched, ignoring class, to the unused ground-truth box it
    overlaps most (IoU >= MATCH_IOU). Returns (accuracy, number of matched predictions).
    """
    test_dir = data_yaml.parent / "test"
    correct = matched = 0
    for image in sorted((test_dir / "images").glob("*.jpg")):
        gt = [ln.split() for ln in (test_dir / "labels" / (image.stem + ".txt")).read_text().splitlines()]
        if not gt:
            continue
        pred = model.predict(str(image), imgsz=imgsz, conf=CLASS_ACC_CONF, verbose=False)[0]
        if len(pred.boxes) == 0:
            continue

        gt_cls = [int(t[0]) for t in gt]
        gt_xywh = torch.tensor([[float(x) for x in t[1:]] for t in gt])
        # normalised centre/size -> pixel corner boxes
        gt_xyxy = torch.cat([gt_xywh[:, :2] - gt_xywh[:, 2:] / 2, gt_xywh[:, :2] + gt_xywh[:, 2:] / 2], 1) * imgsz
        pred_cls = pred.boxes.cls.cpu().int()
        pred_conf = pred.boxes.conf.cpu()
        overlap = box_iou(pred.boxes.xyxy.cpu(), gt_xyxy)

        used: set[int] = set()
        for i in pred_conf.argsort(descending=True).tolist():
            row = overlap[i].clone()
            for j in used:  # each ground-truth box can be matched once
                row[j] = -1
            j = int(row.argmax())
            if row[j] >= MATCH_IOU:
                used.add(j)
                matched += 1
                correct += int(pred_cls[i]) == gt_cls[j]
    return correct / max(matched, 1), matched


def main() -> None:
    a = parse_args()
    tag = f"{a.dataset}_{a.task}_{a.model}_f{a.fold}_s{a.seed}"
    result_path = RESULTS / "sr" / f"{tag}.json"
    if result_path.exists():
        raise SystemExit("skip " + tag)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    data_yaml = DATA / "sr" / a.dataset / a.task / f"fold{a.fold}" / "data.yaml"
    names = yaml.safe_load(data_yaml.read_text())["names"]
    model_cls = RTDETR if a.model.startswith("rtdetr") else YOLO

    # --- train: COCO-pretrained weights, fine-tuned for a fixed number of epochs -------------------------
    t0 = time.time()
    model_cls(f"{a.model}.pt").train(
        data=str(data_yaml),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        seed=a.seed,
        deterministic=True,
        val=False,  # no per-epoch validation -> no "best epoch" picked on test data
        patience=a.epochs,  # disable early stopping
        amp=True,
        plots=False,
        verbose=False,
        workers=4,
        project=str(ROOT / "runs_sr"),
        name=tag,
        exist_ok=True,
    )
    train_min = (time.time() - t0) / 60

    # --- evaluate the final-epoch weights on the held-out block --------------------------------------------
    model = model_cls(str(ROOT / "runs_sr" / tag / "weights" / "last.pt"))
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=a.imgsz, batch=8, conf=0.001, plots=False, verbose=False
    )
    label_dir = data_yaml.parent / "test" / "labels"
    n_instances = count_test_instances(label_dir, len(names))
    # Per-class AP only exists for classes present in the test block (absent classes are skipped by Ultralytics).
    per_class = {
        names[int(c)]: {"map50": float(metrics.box.all_ap[i, 0]), "map50_95": float(metrics.box.all_ap[i].mean())}
        for i, c in enumerate(metrics.box.ap_class_index)
    }
    out = {
        "dataset": a.dataset,
        "task": a.task,
        "model": a.model,
        "fold": a.fold,
        "seed": a.seed,
        "epochs": a.epochs,
        "batch": a.batch,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "train_min": train_min,
        "per_class": per_class,
        "test_instances": dict(zip(names, n_instances, strict=True)),
        "n_test_images": len(list(label_dir.glob("*.txt"))),
    }
    if a.task in ("gender", "posture"):  # tasks where the class is a property of the person
        accuracy, matched = matched_class_accuracy(model, data_yaml, a.imgsz)
        out.update({"class_acc_matched": accuracy, "matched": matched})

    result_path.write_text(json.dumps(out, indent=1))
    shutil.rmtree(ROOT / "runs_sr" / tag, ignore_errors=True)  # checkpoints are not kept (disk)
    print(json.dumps({k: out[k] for k in ["map50", "map50_95", "train_min"]}))


if __name__ == "__main__":
    main()
