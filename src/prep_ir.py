"""Simulated infrared CCTV: build 'gray' and 'irsim' copies of every fold (same labels, same splits).

gray  : luminance only, replicated to 3 channels (keeps COCO-pretrained input shape).
irsim : gray + contrast compression + vignette + gaussian blur + sensor noise (fixed seed per frame).
NOTE: a simulation. Real NIR changes fabric/skin reflectance, which grayscale cannot reproduce.
"""
import os, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "folds"
IMG = ROOT / "data" / "ir_images"

def gray(im): return im.convert("L").convert("RGB")
def irsim(im, seed):
    rng = np.random.default_rng(seed)
    g = np.asarray(im.convert("L"), dtype=np.float32) / 255
    g = 0.15 + 0.7 * g                                   # compressed dynamic range
    h, w = g.shape; yy, xx = np.mgrid[:h, :w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    g = g * (1 - 0.35 * np.clip(r, 0, 1.2) ** 2)          # vignette
    img = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    g = np.asarray(img, dtype=np.float32) / 255 + rng.normal(0, 0.04, g.shape)   # sensor noise
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(g).convert("RGB")

conds = {"gray": lambda im, s: gray(im), "irsim": irsim}
frames = {}
for p in (SRC / "person" / "fold0").rglob("*.jpg"):
    frames[int(p.stem[1:])] = p.resolve()
for i, (cond, fn) in enumerate(conds.items()):
    (IMG / cond).mkdir(parents=True, exist_ok=True)
for f in sorted(SRC.glob("*/fold*/*/images/*.jpg")):
    frames.setdefault(int(f.stem[1:]), f.resolve())
for idx, src in frames.items():
    im = Image.open(src).convert("RGB")
    for cond, fn in conds.items():
        fn(im, idx).save(IMG / cond / f"f{idx:04d}.jpg", quality=95)

for cond in conds:
    out = ROOT / "data" / f"folds_{cond}"
    if out.exists(): shutil.rmtree(out)
    for y in SRC.glob("*/fold*/data.yaml"):
        task, fold = y.parent.parent.name, y.parent.name
        for split in ["train", "test"]:
            d = out / task / fold / split
            (d / "images").mkdir(parents=True); (d / "labels").mkdir(parents=True)
            for im in (y.parent / split / "images").glob("*.jpg"):
                (d / "images" / im.name).symlink_to(IMG / cond / im.name)
            for lb in (y.parent / split / "labels").glob("*.txt"):
                (d / "labels" / lb.name).symlink_to(lb.resolve())
        txt = y.read_text().replace(str(y.parent), str(out / task / fold))
        (out / task / fold / "data.yaml").write_text(txt)
print("ok", len(frames), "frames")
