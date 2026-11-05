import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from aitf_cv.tasks.train_baseline import build_model, pad

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Offline 2-second violence window demo for one local video.")
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
source = args.input.resolve()
out = args.output_dir.resolve()
if not out.is_relative_to((root / "artifacts").resolve()) or out == (root / "artifacts").resolve():
    parser.error("output directory must be inside artifacts/")
out.mkdir(parents=True, exist_ok=True)
video = out / "violence_sliding_2s.mp4"
report_path = out / "predictions.json"
if video.exists() or report_path.exists():
    raise SystemExit("Refusing to overwrite existing output")
threshold = json.loads((root / "artifacts/violence/baseline_v1/metrics.json").read_text())["threshold"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(4)
model = build_model("violence", pretrained=False)
checkpoint = torch.load(root / "artifacts/violence/baseline_v1/best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval().to(device)
cap = cv2.VideoCapture(str(source))
fps = cap.get(cv2.CAP_PROP_FPS)
if not cap.isOpened() or abs(fps - 24) > 0.01:
    raise RuntimeError("Input missing or unexpected frame rate")
window_frames = round(2 * fps)
step_frames = round(0.5 * fps)
buffer = deque(maxlen=window_frames)
font = cv2.FONT_HERSHEY_SIMPLEX


def predict(frames):
    indices = np.linspace(0, len(frames) - 1, 16).astype(int)
    images = []
    for i in indices:
        image = Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        images.append(np.asarray(pad(image, 172), dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(images).transpose(3, 0, 1, 2).copy()).unsqueeze(0).to(device)
    return float(model(x).softmax(dim=1)[0, 1]), [int(i) for i in indices]


def text(frame, value, x, y, size=0.49, color=(255, 255, 255), thick=1):
    cv2.putText(frame, value, (x, y), font, size, color, thick, cv2.LINE_AA)


cmd = [
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-n",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    "960x540",
    "-r",
    str(fps),
    "-i",
    "-",
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(video),
]
encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
index = 0
predictions = []
latest = None
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buffer.append(frame)
        if len(buffer) == window_frames and (index - (window_frames - 1)) % step_frames == 0:
            score, indices = predict(buffer)
            start = index - window_frames + 1
            latest = {
                "window_start_frame": start,
                "window_end_frame": index,
                "window_start_seconds": start / fps,
                "window_end_seconds": (index + 1) / fps,
                "sampled_frame_indices": [start + i for i in indices],
                "violent_score": score,
                "threshold": threshold,
                "violent": score >= threshold,
            }
            predictions.append(latest)
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (535, 148), (30, 28, 24), -1)
        cv2.rectangle(overlay, (8, 501), (952, 532), (30, 28, 24), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        text(frame, "VIOLENCE DETECTION PIPELINE", 20, 32, 0.55, thick=2)
        text(frame, "1  Keep the latest 2 seconds of video", 20, 57)
        text(frame, "2  Pick 16 frames across that time window", 20, 80)
        text(frame, "3  MoViNet classifies the whole window", 20, 103)
        text(frame, "4  Update the result every 0.5 seconds", 20, 126)
        if latest is None:
            cv2.rectangle(frame, (548, 10), (950, 92), (75, 75, 75), -1)
            text(frame, "COLLECTING VIDEO", 563, 39, 0.65, thick=2)
            text(frame, f"{(index + 1) / fps:.1f} / 2.0 seconds", 563, 65, 0.5)
            text(frame, "Waiting for the first complete window", 20, 522, 0.45)
        else:
            positive = latest["violent"]
            color = (28, 22, 180) if positive else (55, 110, 50)
            if positive:
                cv2.rectangle(frame, (0, 0), (959, 539), (0, 0, 240), 8)
            cv2.rectangle(frame, (548, 10), (950, 92), color, -1)
            title = "VIOLENCE SIGNAL" if positive else "NO VIOLENCE SIGNAL"
            text(frame, title, 563, 39, 0.65, thick=2)
            text(frame, f"Window score {latest['violent_score']:.3f}  |  threshold {threshold:.3f}", 563, 65, 0.46)
            text(
                frame,
                f"Window: {latest['window_start_seconds']:.2f}s to {latest['window_end_seconds']:.2f}s | experimental",
                20,
                522,
                0.44,
            )
        encoder.stdin.write(frame.tobytes())
        index += 1
finally:
    cap.release()
    encoder.stdin.close()
    status = encoder.wait()
    if status:
        raise RuntimeError(f"ffmpeg exited {status}")
report = {
    "source": str(source),
    "fps": fps,
    "total_frames": index,
    "window_seconds": 2.0,
    "update_seconds": 0.5,
    "samples_per_window": 16,
    "display": (
        "Window result appears after the last frame and remains until the next update; "
        "full-frame classification, not person localization."
    ),
    "threshold_provenance": "exploratory validation F1 from baseline; not calibrated for two-second synthetic windows",
    "predictions": predictions,
}
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "frames": index,
            "windows": len(predictions),
            "predictions": [(round(x["window_end_seconds"], 2), round(x["violent_score"], 3), x["violent"]) for x in predictions],
        }
    )
)
