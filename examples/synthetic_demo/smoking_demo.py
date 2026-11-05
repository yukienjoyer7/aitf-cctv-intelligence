import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from aitf_cv.tasks.train_baseline import build_model, pad

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Offline per-frame smoking demo for one local video.")
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
video = args.input.resolve()
output_dir = args.output_dir.resolve()
if not output_dir.is_relative_to((root / "artifacts").resolve()) or output_dir == (root / "artifacts").resolve():
    parser.error("output directory must be inside artifacts/")
output_dir.mkdir(parents=True, exist_ok=True)
out = output_dir / "smoking_dynamic.mp4"
if out.exists():
    raise SystemExit("Output already exists")
threshold = json.loads((root / "artifacts/smoking/baseline_v1/metrics.json").read_text())["threshold"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(4)
model = build_model("smoking", pretrained=False)
ckpt = torch.load(root / "artifacts/smoking/baseline_v1/best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval().to(device)
detector = YOLO(str(root / "yolo26s.pt"))
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
font = cv2.FONT_HERSHEY_SIMPLEX


def put(frame, text, x, y, size=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(frame, text, (x, y), font, size, color, thick, cv2.LINE_AA)


cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS)
if not cap.isOpened() or abs(fps - 24) > 0.01:
    raise RuntimeError("Video cannot be opened or has unexpected FPS")
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
    "24",
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
    str(out),
]
enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
frames = 0
alert_frames = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        detections = detector.predict(frame, classes=[0], conf=0.25, device=str(device), verbose=False)[0]
        crops, boxes = [], []
        for x1, y1, x2, y2 in detections.boxes.xyxy.cpu().tolist():
            x1 = max(0, math.floor(x1))
            y1 = max(0, math.floor(y1))
            x2 = min(w, math.ceil(x2))
            y2 = min(h, math.ceil(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            image = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            arr = np.asarray(pad(image, 224), dtype=np.float32) / 255.0
            crops.append(torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1).copy()))
            boxes.append((x1, y1, x2, y2))
        scores = model(torch.stack(crops).to(device)).softmax(dim=1)[:, 1].cpu().tolist() if crops else []
        positive = any(score >= threshold for score in scores)
        alert_frames += int(positive)
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        used_labels = []
        for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
            x1, y1, x2, y2 = (round(x * 0.75) for x in (x1, y1, x2, y2))
            is_positive = score >= threshold
            color = (50, 65, 230) if is_positive else (80, 180, 70)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            text = f"{'SMOKING SIGNAL' if is_positive else 'BELOW THRESHOLD'}  {score:.3f}"
            size = cv2.getTextSize(text, font, 0.43, 1)[0]
            y = max(160, min(485, y1 + 3))
            right = min(959, x1 + size[0] + 12)
            while y < 465 and any(x1 < r and right > left and y < b and y + 20 > t for left, t, r, b in used_labels):
                y += 22
            used_labels.append((x1, y, right, y + 20))
            cv2.rectangle(frame, (x1, y), (right, y + 20), color, -1)
            put(frame, text, x1 + 5, y + 14, 0.43, (255, 255, 255))
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (535, 148), (30, 28, 24), -1)
        cv2.rectangle(overlay, (8, 508), (952, 532), (30, 28, 24), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        put(frame, "SMOKING DETECTION PIPELINE", 20, 32, 0.55, (255, 255, 255), 2)
        put(frame, "1  CCTV frame: take a picture from the video", 20, 57, 0.48)
        put(frame, "2  YOLO26s: find and outline each person", 20, 80, 0.48)
        put(frame, "3  Crop: cut out each detected person", 20, 103, 0.48)
        put(frame, "4  MobileNetV4: check each crop for smoking", 20, 126, 0.48)
        put(frame, f"Per-frame scores | threshold {threshold:.3f} | exploratory, not a confirmed event", 20, 525, 0.43)
        enc.stdin.write(frame.tobytes())
        frames += 1
finally:
    cap.release()
    enc.stdin.close()
    status = enc.wait()
    if status:
        raise RuntimeError(f"ffmpeg exited {status}")
print(json.dumps({"frames": frames, "alert_frames": alert_frames, "threshold": threshold, "device": str(device)}))
