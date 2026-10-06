"""CCTV-SR indoor/outdoor -> leakage-aware blocked 5-fold CV, one folder tree per task.

The Roboflow splits are random over consecutive frames of ONE fixed camera (near-duplicate leakage), so all splits are
pooled, sorted by frame id and cut into K contiguous blocks. Train frames within BUF frame-ids of the held-out block are dropped.
Quad labels are converted to enclosing boxes.
Tasks (class remaps):
  indoor : person (all people), gender (F/M), allclass (3 classes incl. windows)
  outdoor: person, gender (F/M, posture merged), posture (Sitting/Standing/Walking, gender merged), allclass (9 classes)
Images with no relevant boxes are kept as background images.
"""
import re, shutil, json
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "data" / "sr"; K = 5
BUF = {"indoor": 5, "outdoor": 10}
def remap(names):
    idx = {n: i for i, n in enumerate(names)}; people = [n for n in names if n.startswith(("Female_", "Male_"))]
    T = {"person": ({n: 0 for n in people}, ["person"]),
         "gender": ({n: (0 if n.startswith("Female") else 1) for n in people}, ["female", "male"]),
         "allclass": ({n: i for i, n in enumerate(names)}, list(names))}
    posts = sorted({n.split("_")[1] for n in people})
    if len(posts) > 1: T["posture"] = ({n: posts.index(n.split("_")[1]) for n in people}, [p.lower() for p in posts])
    return idx, T
manifest = {}
for ds in ["indoor", "outdoor"]:
    src = ROOT / "data" / ds; names = yaml.safe_load((src / "data.yaml").read_text())["names"]; idx, TASKS = remap(names)
    fr = []
    for sp in ["train", "valid", "test"]:
        for im in (src / sp / "images").glob("*.jpg"):
            fid = int(re.search(r"frame_0*(\d+)", im.name).group(1)); L = []
            for ln in (src / sp / "labels" / (im.stem + ".txt")).read_text().splitlines():
                c, *v = ln.split(); v = list(map(float, v))
                if len(v) > 4: xs, ys = v[0::2], v[1::2]; v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                L.append((names[int(c)], v))
            fr.append((fid, im, L))
    fr.sort(key=lambda t: t[0]); n = len(fr); blocks = np.array_split(np.arange(n), 5); manifest[ds] = {"n": n, "folds": {}}
    for task, (mp, tnames) in TASKS.items():
        t_out = OUT / ds / task
        if t_out.exists(): shutil.rmtree(t_out)
        for k, blk in enumerate(blocks):
            lo, hi = fr[blk[0]][0] - BUF[ds], fr[blk[-1]][0] + BUF[ds]
            test = [fr[i] for i in blk]; train = [f for i, f in enumerate(fr) if not (lo <= f[0] <= hi)]
            manifest[ds]["folds"][k] = {"n_train": len(train), "n_test": len(test), "test_ids": [test[0][0], test[-1][0]]}
            for split, items in [("train", train), ("test", test)]:
                d = t_out / f"fold{k}" / split; (d / "images").mkdir(parents=True); (d / "labels").mkdir(parents=True)
                for fid, im, L in items:
                    (d / "images" / f"f{fid:04d}.jpg").symlink_to(im.resolve())
                    (d / "labels" / f"f{fid:04d}.txt").write_text("".join(f"{mp[nm]} " + " ".join(f"{x:.6f}" for x in v) + "\n" for nm, v in L if nm in mp))
            (t_out / f"fold{k}" / "data.yaml").write_text(f"path: {t_out / f'fold{k}'}\ntrain: train/images\nval: test/images\ntest: test/images\nnc: {len(tnames)}\nnames: {tnames}\n")
    print(ds, n, "frames; tasks:", list(TASKS), "; folds (train/test):", [(v["n_train"], v["n_test"]) for v in manifest[ds]["folds"].values()])
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
