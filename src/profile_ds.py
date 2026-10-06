import sys, re, hashlib, collections as C
from pathlib import Path
import numpy as np, yaml
from PIL import Image
d = Path(sys.argv[1]); names = yaml.safe_load((d / "data.yaml").read_text())["names"]
def src(n): return re.sub(r"_(jpg|png|jpeg)\.rf\..*$", "", n)  # roboflow suffix
rows = []  # split, stem, cls, w, h, poly
per_img = {}
for sp in ["train", "valid", "test"]:
    for im in sorted((d / sp / "images").glob("*.*")):
        lb = d / sp / "labels" / (im.stem + ".txt"); L = lb.read_text().splitlines() if lb.exists() else []
        per_img[(sp, im.name)] = len(L)
        for ln in L:
            c, *v = ln.split(); v = list(map(float, v)); poly = len(v) > 4
            if poly: xs, ys = v[0::2], v[1::2]; w, h = max(xs) - min(xs), max(ys) - min(ys)
            else: w, h = v[2], v[3]
            rows.append((sp, im.name, int(c), w, h, poly))
print("images/split:", {s: sum(1 for k in per_img if k[0] == s) for s in ["train", "valid", "test"]}, "| boxes:", len(rows), "| polygons:", sum(r[5] for r in rows))
print("empty images:", sum(1 for v in per_img.values() if v == 0))
cnt = C.Counter((r[0], r[2]) for r in rows)
print(f"\n{'class':20s} {'train':>6s} {'valid':>6s} {'test':>6s} {'total':>6s}")
for i, n in enumerate(names):
    t = [cnt[(s, i)] for s in ["train", "valid", "test"]]; print(f"{n:20s} {t[0]:6d} {t[1]:6d} {t[2]:6d} {sum(t):6d}")
bpi = np.array(list(per_img.values())); print(f"\nboxes/image: mean {bpi.mean():.1f}, median {np.median(bpi):.0f}, min {bpi.min()}, max {bpi.max()}")
w = np.array([r[3] for r in rows]) * 512; h = np.array([r[4] for r in rows]) * 512; a = np.sqrt(w * h)
print(f"box size px (512 img): sqrt-area median {np.median(a):.0f}, p5 {np.percentile(a,5):.0f}, p95 {np.percentile(a,95):.0f}; <16px: {(a<16).mean()*100:.1f}%; <32px: {(a<32).mean()*100:.1f}%")
print("per-class median sqrt-area px:", {names[i]: int(np.median(a[[r[2] == i for r in rows]])) for i in range(len(names)) if any(r[2] == i for r in rows)})
# sources / leakage
S = C.defaultdict(C.Counter)
for (sp, n) in per_img: S[src(n).rsplit("-", 1)[0] if "-" in n else re.sub(r"\d+$", "", src(n))][sp] += 1
print("\nimage name prefixes (source) by split:"); [print(" ", k[:70], dict(v)) for k, v in sorted(S.items(), key=lambda x: -sum(x[1].values()))[:12]]
# exact duplicate / cross-split near duplicate by tiny hash
def h8(p):
    im = Image.open(p).convert("L").resize((16, 16)); x = np.asarray(im, float); return (x > x.mean()).flatten()
H = {k: h8(d / k[0] / "images" / k[1]) for k in per_img}
keys = list(H); tr = [k for k in keys if k[0] == "train"]; Tm = np.array([H[k] for k in tr])
for sp in ["valid", "test"]:
    ds = [(Tm != H[k]).sum(1).min() for k in keys if k[0] == sp]; ds = np.array(ds)
    print(f"{sp}: min Hamming (256-bit avg-hash) to nearest TRAIN image: median {np.median(ds):.0f}, frames with <=10 bits: {(ds<=10).mean()*100:.0f}%, <=20: {(ds<=20).mean()*100:.0f}%")
