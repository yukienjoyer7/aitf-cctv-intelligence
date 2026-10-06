"""Train one (dataset, task, model, fold, seed) on the CCTV-SR blocked folds; evaluate FINAL-epoch weights on the held-out block.
Threshold-free AP only; class-match accuracy (gender/posture tasks) uses a fixed a-priori conf=0.25. Checkpoints deleted after eval."""
import argparse, json, shutil, time
from pathlib import Path
import torch, yaml
from torchvision.ops import box_iou
from ultralytics import YOLO, RTDETR

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
for k in ["dataset", "task", "model"]: ap.add_argument("--" + k, required=True)
ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, required=True); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--imgsz", type=int, default=512)
a = ap.parse_args()
tag = f"{a.dataset}_{a.task}_{a.model}_f{a.fold}_s{a.seed}"; res = ROOT / "results" / "sr" / f"{tag}.json"
if res.exists(): raise SystemExit("skip " + tag)
res.parent.mkdir(parents=True, exist_ok=True)
data = ROOT / "data" / "sr" / a.dataset / a.task / f"fold{a.fold}" / "data.yaml"; names = yaml.safe_load(data.read_text())["names"]
cls = RTDETR if a.model.startswith("rtdetr") else YOLO
t0 = time.time()
cls(f"{a.model}.pt").train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=True, val=False, plots=False,
                           workers=4, project=str(ROOT / "runs_sr"), name=tag, exist_ok=True, patience=a.epochs, verbose=False, amp=True)
train_min = (time.time() - t0) / 60
m = cls(str(ROOT / "runs_sr" / tag / "weights" / "last.pt"))
v = m.val(data=str(data), split="test", imgsz=a.imgsz, batch=8, conf=0.001, plots=False, verbose=False)
lab_dir = data.parent / "test" / "labels"; n_inst = [0] * len(names)
for p in lab_dir.glob("*.txt"):
    for ln in p.read_text().splitlines(): n_inst[int(ln.split()[0])] += 1
pc = {names[int(c)]: {"map50": float(v.box.all_ap[i, 0]), "map50_95": float(v.box.all_ap[i].mean())} for i, c in enumerate(v.box.ap_class_index)}
out = {"dataset": a.dataset, "task": a.task, "model": a.model, "fold": a.fold, "seed": a.seed, "epochs": a.epochs, "batch": a.batch,
       "map50": float(v.box.map50), "map50_95": float(v.box.map), "train_min": train_min, "per_class": pc,
       "test_instances": dict(zip(names, n_inst)), "n_test_images": len(list(lab_dir.glob("*.txt")))}
if a.task in ("gender", "posture"):   # class accuracy on localised people (class-agnostic IoU>=0.5 match, fixed conf 0.25)
    ok = tot = 0
    for p in sorted((data.parent / "test" / "images").glob("*.jpg")):
        gt = [ln.split() for ln in (lab_dir / (p.stem + ".txt")).read_text().splitlines()]
        r = m.predict(str(p), imgsz=a.imgsz, conf=0.25, verbose=False)[0]
        if not gt or len(r.boxes) == 0: continue
        g = torch.tensor([[float(x) for x in t[1:]] for t in gt]); gc = [int(t[0]) for t in gt]
        gx = torch.cat([g[:, :2] - g[:, 2:] / 2, g[:, :2] + g[:, 2:] / 2], 1) * a.imgsz
        pb, pcl, ps = r.boxes.xyxy.cpu(), r.boxes.cls.cpu().int(), r.boxes.conf.cpu(); iou = box_iou(pb, gx); used = set()
        for i in ps.argsort(descending=True).tolist():
            row = iou[i].clone()
            for j in used: row[j] = -1
            j = int(row.argmax())
            if row[j] >= 0.5: used.add(j); tot += 1; ok += int(pcl[i]) == gc[j]
    out.update({"class_acc_matched": ok / max(tot, 1), "matched": tot})
res.write_text(json.dumps(out, indent=1)); shutil.rmtree(ROOT / "runs_sr" / tag, ignore_errors=True)
print(json.dumps({k: out[k] for k in ["map50", "map50_95", "train_min"]}))
