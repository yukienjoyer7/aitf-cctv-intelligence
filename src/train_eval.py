"""Train one (model, task, fold, seed) and evaluate the FINAL-epoch weights on the held-out block.

No checkpoint selection / threshold tuning on the held-out data. Only threshold-free metrics
(AP) are taken from Ultralytics; gender accuracy uses a fixed a-priori conf=0.25.
"""
import argparse, json, time
from pathlib import Path
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO, RTDETR

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)  # yolo26n | yolo26s | rtdetr-l
ap.add_argument("--task", required=True)   # person | gender
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()

tag = f"{a.model}_{a.task}_f{a.fold}_s{a.seed}"
res_path = ROOT / "results" / "runs" / f"{tag}.json"
if res_path.exists():
    raise SystemExit(f"skip {tag}")
res_path.parent.mkdir(parents=True, exist_ok=True)
data = ROOT / "data" / "folds" / a.task / f"fold{a.fold}" / "data.yaml"
cls = RTDETR if a.model.startswith("rtdetr") else YOLO
model = cls(f"{a.model}.pt")
t0 = time.time()
model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=True,
            val=False, plots=False, workers=4, project=str(ROOT / "runs"), name=tag, exist_ok=True,
            patience=a.epochs, verbose=False, amp=True)
train_min = (time.time() - t0) / 60
last = ROOT / "runs" / tag / "weights" / "last.pt"
m = cls(str(last))
v = m.val(data=str(data), split="test", imgsz=a.imgsz, batch=8, conf=0.001, plots=False, verbose=False)
out = {"model": a.model, "task": a.task, "fold": a.fold, "seed": a.seed, "epochs": a.epochs,
       "map50": float(v.box.map50), "map50_95": float(v.box.map), "train_min": train_min,
       "per_class_map50_95": {int(c): float(x) for c, x in zip(v.box.ap_class_index, v.box.maps[v.box.ap_class_index])}}

# gender accuracy on localised persons at a fixed, a-priori conf threshold
if a.task == "gender":
    img_dir = data.parent / "test" / "images"
    ok = tot_match = n_gt = 0
    for p in sorted(img_dir.glob("*.jpg")):
        r = m.predict(str(p), imgsz=a.imgsz, conf=0.25, verbose=False)[0]
        gt = [ln.split() for ln in (data.parent / "test" / "labels" / (p.stem + ".txt")).read_text().splitlines()]
        g = torch.tensor([[float(x) for x in t[1:]] for t in gt]); gc = [int(t[0]) for t in gt]
        n_gt += len(gc)
        if len(r.boxes) == 0: continue
        gxy = torch.cat([g[:, :2] - g[:, 2:] / 2, g[:, :2] + g[:, 2:] / 2], 1) * a.imgsz
        pb, pc, ps = r.boxes.xyxy.cpu(), r.boxes.cls.cpu().int(), r.boxes.conf.cpu()
        iou = box_iou(pb, gxy); used = set()
        for i in ps.argsort(descending=True).tolist():
            row = iou[i].clone()
            for j in used: row[j] = -1
            j = int(row.argmax())
            if row[j] >= 0.5:
                used.add(j); tot_match += 1; ok += int(pc[i]) == gc[j]
    out.update({"gender_acc_matched": ok / max(tot_match, 1), "matched": tot_match, "n_gt": n_gt})
res_path.write_text(json.dumps(out, indent=1))
if not (a.fold == 0 and a.seed == 0 and a.task == "gender"):  # keep only weights needed for latency; disk is tight
    import shutil; shutil.rmtree(ROOT / "runs" / tag, ignore_errors=True)
print(json.dumps(out))
