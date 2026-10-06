"""Latency / complexity benchmark (batch 1, 512x512). Reports model-only and end-to-end (pre+infer+post)."""
import json, time, glob, statistics as st, sys
from pathlib import Path
import numpy as np, torch, cv2
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
SZ = 512
img = cv2.imread(sorted(glob.glob(str(ROOT / "data/folds/gender/fold0/test/images/*.jpg")))[0])
img = cv2.resize(img, (SZ, SZ))

def stats(ts):
    ts = sorted(ts); n = len(ts)
    return {"mean": st.mean(ts), "std": st.pstdev(ts), "median": ts[n // 2], "p95": ts[int(n * .95) - 1], "n": n}

def timeit(fn, n, warm, cuda):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        if cuda: torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if cuda: torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return stats(ts)

out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "imgsz": SZ, "models": {}}
for name in ["yolo26n", "yolo26s", "rtdetr-l"]:
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    w = ROOT / "runs" / f"{name}_gender_f0_s0" / "weights" / "last.pt"
    y = cls(str(w if w.exists() else f"{name}.pt"))
    params = sum(p.numel() for p in y.model.parameters())
    r = {"params_M": params / 1e6, "gflops": get_flops(y.model, SZ)}
    for dev, half, n, warm in [("cuda", False, 300, 50), ("cuda", True, 300, 50), ("cpu", False, 40, 5)]:
        key = f"{dev}_{'fp16' if half else 'fp32'}"
        if dev == "cpu": torch.set_num_threads(8)
        y = cls(str(w if w.exists() else f"{name}.pt")); y.to(dev)
        y.fuse(); net = y.model.to(dev).eval()  # fuse conv+bn, as predict() does
        if half: net = net.half()
        x = torch.rand(1, 3, SZ, SZ, device=dev, dtype=torch.half if half else torch.float32)
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            mo = timeit(lambda: net(x), n, warm, dev == "cuda")
            e2e = timeit(lambda: y.predict(img, imgsz=SZ, conf=0.25, device=dev, half=half, verbose=False), n, warm, dev == "cuda")
        r[key] = {"model_only_ms": mo, "end_to_end_ms": e2e, "fps_e2e": 1e3 / e2e["mean"]}
        if dev == "cuda": r[key]["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
        print(name, key, round(mo["mean"], 2), round(e2e["mean"], 2), flush=True)
    out["models"][name] = r
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=1))
