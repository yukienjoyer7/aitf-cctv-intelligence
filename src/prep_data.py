"""Build leakage-aware blocked cross-validation folds from the Roboflow export.

* Merges the original train/valid/test splits (frames come from ONE 2-minute video,
  ~1 s apart, so the random Roboflow split leaks near-duplicate frames).
* Fixes 3 frames whose boxes were labelled with the stray 'AITF' class.
* Splits by contiguous time blocks (K folds) with a buffer of frames removed from train
  on each side of the held-out block.
* Emits two tasks: `person` (1 class) and `gender` (Female/Male).
"""
import re, shutil, zipfile, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "folds"
K, BUFFER = 5, 4

if not RAW.exists():
    with zipfile.ZipFile(ROOT / "data" / "AITF.v1i.yolo26.zip") as z:
        z.extractall(RAW)

frames = {}
for split in ["train", "valid", "test"]:
    for img in (RAW / split / "images").glob("*.jpg"):
        idx = int(re.search(r"mp4-(\d+)_jpg", img.name).group(1))
        frames[idx] = (split, img, RAW / split / "labels" / (img.stem + ".txt"))
order = sorted(frames)
assert len(order) == 120

# label fix: class 0 ('AITF') -> Male in train frame 10 (all boys), -> Female in valid 20 / test 35 (hijab girls)
FIX = {10: 2, 20: 1, 35: 1}
def load(idx):
    rows = []
    for ln in frames[idx][2].read_text().splitlines():
        c, *b = ln.split()
        c = int(c); b = list(map(float, b))
        if len(b) > 4:  # polygon (118 quads + 1 23-gon in the export) -> enclosing box in YOLO xywh
            xs, ys = b[0::2], b[1::2]
            b = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
        if c == 0:
            c = FIX[idx]
        rows.append((c, *b))
    return rows
labels = {i: load(i) for i in order}
assert all(r[0] in (1, 2) for i in order for r in labels[i])
stats = {"frames": len(order), "female": sum(r[0] == 1 for i in order for r in labels[i]),
         "male": sum(r[0] == 2 for i in order for r in labels[i])}

blocks = np.array_split(np.arange(len(order)), K)
if OUT.exists():
    shutil.rmtree(OUT)
def write(task, fold, split, ids):
    d = OUT / task / f"fold{fold}" / split
    (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
    for i in ids:
        src = frames[i][1]; name = f"f{i:04d}"
        (d / "images" / f"{name}.jpg").symlink_to(src)
        with open(d / "labels" / f"{name}.txt", "w") as f:
            for c, *b in labels[i]:
                cls = 0 if task == "person" else c - 1
                f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in b) + "\n")
manifest = {}
for k, blk in enumerate(blocks):
    test = [order[j] for j in blk]
    lo, hi = blk[0] - BUFFER, blk[-1] + BUFFER
    train = [order[j] for j in range(len(order)) if j < lo or j > hi]
    manifest[k] = {"test": test, "train": train, "n_train": len(train), "n_test": len(test)}
    for task, names in [("person", ["person"]), ("gender", ["female", "male"])]:
        write(task, k, "train", train); write(task, k, "test", test)
        yaml = OUT / task / f"fold{k}" / "data.yaml"
        yaml.write_text(f"path: {yaml.parent}\ntrain: train/images\nval: test/images\ntest: test/images\n"
                        f"nc: {len(names)}\nnames: {names}\n")
(OUT / "manifest.json").write_text(json.dumps({"stats": stats, "folds": manifest}, indent=1))
print(stats); print({k: (v['n_train'], v['n_test']) for k, v in manifest.items()})
