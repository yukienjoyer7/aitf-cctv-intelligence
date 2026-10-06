"""Latency and complexity benchmark: batch 1, 512x512, for YOLO26n / YOLO26s / RT-DETR-L.

For each model and setting (GPU fp32, GPU fp16, CPU fp32) it reports, over many timed iterations after a warm-up:
  * model-only time: just the network forward pass (conv+BN fused, as Ultralytics' predict() does);
  * end-to-end time: preprocessing + forward pass + postprocessing (NMS for YOLO, none for RT-DETR) via predict();
  * mean / std / median / p95, FPS, peak GPU memory, parameters and GFLOPs.
GPU timings call cuda.synchronize() around every iteration so queued kernels are included.

Provenance note: results/latency.json was produced by an earlier revision of this script that loaded fold-0
weights from a pilot experiment (2-class head) and a pilot frame. Latency depends on the architecture, not on the
trained weight values, so this version uses the pretrained COCO checkpoints and a CCTV-SR frame instead; the
numbers are expected to agree up to run-to-run noise (laptop GPU, so thermal throttling is possible).

Usage: python src/latency.py        (needs a GPU; run with nothing else using it)
"""

import glob
import json
import statistics
import time

import cv2
import torch
from ultralytics import RTDETR, YOLO
from ultralytics.utils.torch_utils import get_flops

from common import DATA, IMG_SIZE, MODELS, RESULTS

# (device, half precision, timed iterations, warm-up iterations); CPU is slow, so fewer iterations
SETTINGS = [("cuda", False, 300, 50), ("cuda", True, 300, 50), ("cpu", False, 40, 5)]
CPU_THREADS = 8


def summarise(times_ms: list[float]) -> dict:
    times = sorted(times_ms)
    n = len(times)
    return {
        "mean": statistics.mean(times),
        "std": statistics.pstdev(times),
        "median": times[n // 2],
        "p95": times[int(n * 0.95) - 1],
        "n": n,
    }


def time_it(fn, n: int, warmup: int, cuda: bool) -> dict:
    """Run `fn` `warmup` times untimed, then time `n` calls in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()  # make sure earlier GPU work is finished before starting the clock
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()  # ... and that this call's GPU work is finished before stopping it
        times.append((time.perf_counter() - start) * 1e3)
    return summarise(times)


def benchmark_setting(model_cls, name: str, frame, device: str, half: bool, n: int, warmup: int) -> dict:
    """Time one model in one setting (fresh model instance so settings do not affect each other)."""
    wrapper = model_cls(f"{name}.pt")
    wrapper.to(device)
    wrapper.fuse()  # fuse conv+BN, as predict() does
    net = wrapper.model.to(device).eval()
    if half:
        net = net.half()
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device, dtype=torch.half if half else torch.float32)
    cuda = device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        model_only = time_it(lambda: net(x), n, warmup, cuda)
        end_to_end = time_it(
            lambda: wrapper.predict(frame, imgsz=IMG_SIZE, conf=0.25, device=device, half=half, verbose=False), n, warmup, cuda
        )
    result = {"model_only_ms": model_only, "end_to_end_ms": end_to_end, "fps_e2e": 1e3 / end_to_end["mean"]}
    if cuda:
        result["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    return result


def main() -> None:
    frame_path = sorted(glob.glob(str(DATA / "sr/indoor/person/fold0/test/images/*.jpg")))[0]
    frame = cv2.resize(cv2.imread(frame_path), (IMG_SIZE, IMG_SIZE))
    torch.set_num_threads(CPU_THREADS)

    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "imgsz": IMG_SIZE, "models": {}}
    for name in MODELS:
        model_cls = RTDETR if name.startswith("rtdetr") else YOLO
        reference = model_cls(f"{name}.pt")
        entry = {
            "params_M": sum(p.numel() for p in reference.model.parameters()) / 1e6,
            "gflops": get_flops(reference.model, IMG_SIZE),
        }
        for device, half, n, warmup in SETTINGS:
            key = f"{device}_{'fp16' if half else 'fp32'}"
            entry[key] = benchmark_setting(model_cls, name, frame, device, half, n, warmup)
            print(
                name,
                key,
                round(entry[key]["model_only_ms"]["mean"], 2),
                round(entry[key]["end_to_end_ms"]["mean"], 2),
                flush=True,
            )
        out["models"][name] = entry

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
