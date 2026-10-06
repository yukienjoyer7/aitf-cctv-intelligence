"""YOLO26n on the predefined-split fisheye dataset (train=Lunch1+2, valid=Lunch3, test=Edge_Cases video).
Polygons (5/10-point) and plain boxes are converted to enclosing YOLO boxes. Final-epoch weights, no checkpoint selection."""
import json, shutil, sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "data" / "fisheye"; DST = ROOT / "data" / "fisheye_boxes"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if DST.exists(): shutil.rmtree(DST)
n_poly = n_box = 0
for sp in ["train", "valid", "test"]:
    (DST / sp / "images").mkdir(parents=True); (DST / sp / "labels").mkdir(parents=True)
    for im in (SRC / sp / "images").glob("*.jpg"):
        (DST / sp / "images" / im.name).symlink_to(im.resolve())
        out = []
        for ln in (SRC / sp / "labels" / (im.stem + ".txt")).read_text().splitlines():
            c, *v = ln.split(); v = list(map(float, v))
            if len(v) > 4:
                xs, ys = v[0::2], v[1::2]; n_poly += 1
                v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            else: n_box += 1
            out.append(f"{c} " + " ".join(f"{x:.6f}" for x in v))
        (DST / sp / "labels" / (im.stem + ".txt")).write_text("\n".join(out) + "\n")
(DST / "data.yaml").write_text(f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: ['door_open', 'person_male']\n")
print("polygons", n_poly, "boxes", n_box, flush=True)
tag = f"fisheye_yolo26n_s{SEED}"
if not (ROOT / "runs_fish" / tag / "weights" / "last.pt").exists():
  YOLO("yolo26n.pt").train(data=str(DST / "data.yaml"), epochs=100, imgsz=512, batch=8, seed=SEED, deterministic=True, val=False, plots=False,
                         workers=4, project=str(ROOT / "runs_fish"), name=tag, exist_ok=True, patience=100, verbose=False, amp=True)
m = YOLO(str(ROOT / "runs_fish" / tag / "weights" / "last.pt")); res = {"seed": SEED}
for split in ["val", "test"]:  # Ultralytics names the validation split "val"
    v = m.val(data=str(DST / "data.yaml"), split=split, imgsz=512, batch=8, conf=0.001, plots=False, verbose=False)
    pc = {int(c): {"map50": float(v.box.all_ap[i, 0]), "map50_95": float(v.box.all_ap[i].mean())} for i, c in enumerate(v.box.ap_class_index)}
    res["valid" if split == "val" else split] = {"all_classes_present": {"map50": float(v.box.map50), "map50_95": float(v.box.map)}, "per_class": pc, "class_ids_evaluated": [int(c) for c in v.box.ap_class_index]}
(ROOT / "results").mkdir(exist_ok=True); (ROOT / "results" / f"fisheye_yolo26n_s{SEED}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
