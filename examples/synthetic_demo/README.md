# Offline synthetic-video demos

These scripts reproduce the exploratory visual demos on one local 24 FPS video each. They do not implement a streaming service or event aggregation. No datasets, clips, checkpoints, thresholds, or generated videos are included in Git.

Install the benchmark and optional training requirements into `.venv`. The violence demo requires the pinned MoViNet implementation from `requirements-training.txt`. Use FFmpeg with libx264. Put the local model files at `yolo26s.pt`, `artifacts/smoking/baseline_v1/best.pt`, and `artifacts/violence/baseline_v1/best.pt`; each task also reads its local `baseline_v1/metrics.json` for the validation-selected threshold. These files must come from an authorized source; neither weights nor metrics are downloaded by these demos.

From the repository root, choose input files and new output directories inside the Git-ignored `artifacts/` directory:

```bash
PYTHONPATH=src .venv/bin/python examples/synthetic_demo/smoking_demo.py \
  --input data/synthetic/drive_20261105/merokok/M01.mp4 \
  --output-dir artifacts/demo_smoking_new

PYTHONPATH=src .venv/bin/python examples/synthetic_demo/violence_window_demo.py \
  --input data/synthetic/drive_20261105/kekerasan/K01_indoor.mp4 \
  --output-dir artifacts/demo_violence_new
```

Smoking detects people with YOLO26s and classifies each detected crop on each frame. The bounding box is a person box, not a cigarette box. Violence takes a full-frame 2-second window, samples 16 frames, and updates the displayed window prediction every 0.5 seconds. The violence script writes the sampled indices and scores to `predictions.json` beside the video. The result becomes available after the full window and stays on screen until the next update; it does not locate a violent person or pinpoint the start of an event.

Thresholds were selected on the exploratory public validation data, not calibrated for these synthetic videos or 2-second windows. Scores are model outputs, not verified events. A negative or missed person detection does not establish that no smoking occurred. These videos are controlled stress tests, not evidence of field reliability. Preserve privacy and permissions for all local media.
