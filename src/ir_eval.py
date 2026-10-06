"""YOLO26n infrared-simulation experiment for one (task, fold, seed).
Train on each condition {rgb, gray, irsim}; evaluate final-epoch weights on the held-out block in ALL three conditions."""
import argparse, json, shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=100)
a = ap.parse_args()
DIRS = {"rgb": "folds", "gray": "folds_gray", "irsim": "folds_irsim"}
res = ROOT / "results" / "ir" / f"yolo26n_{a.task}_f{a.fold}_s{a.seed}.json"
if res.exists(): raise SystemExit("skip")
res.parent.mkdir(parents=True, exist_ok=True)
data = lambda c: str(ROOT / "data" / DIRS[c] / a.task / f"fold{a.fold}" / "data.yaml")
out = {"task": a.task, "fold": a.fold, "seed": a.seed, "results": {}}
for tr in DIRS:
    tag = f"ir_{tr}_{a.task}_f{a.fold}_s{a.seed}"
    YOLO("yolo26n.pt").train(data=data(tr), epochs=a.epochs, imgsz=512, batch=8, seed=a.seed, deterministic=True, val=False,
                             plots=False, workers=4, project=str(ROOT / "runs_ir"), name=tag, exist_ok=True,
                             patience=a.epochs, verbose=False, amp=True)
    m = YOLO(str(ROOT / "runs_ir" / tag / "weights" / "last.pt"))
    for te in DIRS:
        v = m.val(data=data(te), split="test", imgsz=512, batch=8, conf=0.001, plots=False, verbose=False)
        out["results"][f"train_{tr}__test_{te}"] = {"map50": float(v.box.map50), "map50_95": float(v.box.map)}
    shutil.rmtree(ROOT / "runs_ir" / tag, ignore_errors=True)
res.write_text(json.dumps(out, indent=1)); print(json.dumps(out))
