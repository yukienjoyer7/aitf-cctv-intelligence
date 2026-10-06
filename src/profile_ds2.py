import sys, re, collections as C
from pathlib import Path
import numpy as np, yaml
from PIL import Image
d = Path(sys.argv[1]); names = yaml.safe_load((d / "data.yaml").read_text())["names"]
PEOPLE = [i for i, n in enumerate(names) if re.match(r"(Fe)?[Mm]ale_", n)]; STATIC = [i for i in range(len(names)) if i not in PEOPLE]
def boxes(sp, im):
    out = []
    for ln in (d / sp / "labels" / (im.stem + ".txt")).read_text().splitlines():
        c, *v = ln.split(); v = list(map(float, v))
        if len(v) > 4: xs, ys = v[0::2], v[1::2]; b = (min(xs), min(ys), max(xs), max(ys))
        else: b = (v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2)
        out.append((int(c), b))
    return out
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-12)
D = {}; fid = {}
for sp in ["train", "valid", "test"]:
    for im in sorted((d / sp / "images").glob("*.jpg")): D[(sp, im.name)] = (im, boxes(sp, im)); fid[(sp, im.name)] = int(re.search(r"frame_0*(\d+)", im.name).group(1))
ids = sorted(fid.values()); print("frame id range", ids[0], "-", ids[-1], "| distinct ids", len(set(ids)), "of", len(ids), "| id gaps >20:", [(a, b) for a, b in zip(ids, ids[1:]) if b - a > 20])
pp = {k: sum(c in PEOPLE for c, _ in v[1]) for k, v in D.items()}
for sp in ["train", "valid", "test"]:
    x = np.array([v for k, v in pp.items() if k[0] == sp]); print(f"{sp}: people/img mean {x.mean():.1f}, images with 0 people {np.mean(x==0)*100:.0f}%, max {x.max()}")
tot = sum(len(v[1]) for v in D.values()); st = sum(c in STATIC for v in D.values() for c, _ in v[1]); print(f"static (window/door) boxes: {st}/{tot} = {st/tot*100:.0f}% of all boxes; people boxes: {tot-st}")
fem = sum(c in PEOPLE and 'Female' in names[c] for v in D.values() for c, _ in v[1]); print(f"people: Female {fem}, Male {tot-st-fem}  ({fem/(tot-st)*100:.0f}% / {(tot-st-fem)/(tot-st)*100:.0f}%)")
# static-label leakage: fraction of valid/test STATIC boxes that have a same-class train box at IoU>=0.7 (i.e. same pixel position)
tr = [(c, b) for k, v in D.items() if k[0] == "train" for c, b in v[1] if c in STATIC]
for sp in ["valid", "test"]:
    q = [(c, b) for k, v in D.items() if k[0] == sp for c, b in v[1] if c in STATIC]
    hit = np.mean([any(c == c2 and iou(b, b2) >= 0.7 for c2, b2 in tr) for c, b in q]); print(f"{sp}: {hit*100:.0f}% of static boxes have a same-class TRAIN box with IoU>=0.7 (n={len(q)})")
# pixel closeness to train
def g(k): return np.asarray(Image.open(D[k][0]).convert("L").resize((128, 128)), np.float32)
G = {k: g(k) for k in D}; trk = [k for k in D if k[0] == "train"]; T = np.stack([G[k] for k in trk])
for sp in ["valid", "test"]:
    m = []; near = []
    for k in [k for k in D if k[0] == sp]:
        dd = np.abs(T - G[k]).mean((1, 2)); j = dd.argmin(); m.append(dd[j]); near.append(abs(fid[trk[j]] - fid[k]))
    print(f"{sp}: mean abs pixel diff (0-255) to nearest train frame: median {np.median(m):.1f}; nearest-train frame id distance: median {np.median(near):.0f}, within 5 ids: {np.mean(np.array(near)<=5)*100:.0f}%")
# reference: diff between two random train frames
rng = np.random.default_rng(0); a = rng.choice(len(trk), 200); b = rng.choice(len(trk), 200); print("reference: random train-vs-train mean abs diff median", round(float(np.median([np.abs(T[i] - T[j]).mean() for i, j in zip(a, b) if i != j])), 1))
