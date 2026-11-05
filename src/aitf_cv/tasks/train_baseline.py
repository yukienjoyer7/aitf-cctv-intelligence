"""Train exploratory smoking or violence baselines from the existing grouped manifests."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
LABELS = {"smoking": ["non_smoking", "smoking"], "violence": ["non_violent", "violent"]}


def read_manifest(task, split):
    path = ROOT / "data/splits" / f"{task}_v1" / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        if row["label"] not in LABELS[task] or not (ROOT / row["path"]).is_file():
            raise ValueError(f"Invalid label or missing file: {row[path]}")
    return rows


def validate_splits(task):
    data = {split: read_manifest(task, split) for split in ("train", "val", "test")}
    for i, a in enumerate(data):
        for b in list(data)[i + 1 :]:
            if {r["group_id"] for r in data[a]} & {r["group_id"] for r in data[b]}:
                raise ValueError(f"Groups overlap between {a} and {b}")
    return data


def pad(image, size):
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    result = Image.new("RGB", (size, size), (0, 0, 0))
    result.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return result


class Samples(Dataset):
    def __init__(self, task, rows, training=False):
        self.task, self.rows, self.training = task, rows, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = ROOT / row["path"]
        if self.task == "smoking":
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
            img = pad(img, 224)
            if self.training and random.random() < 0.5:
                img = ImageOps.mirror(img)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        else:
            cap = cv2.VideoCapture(str(path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count < 1:
                raise ValueError(f"Unreadable video: {path}")
            frames = []
            try:
                for frame_index in np.linspace(0, count - 1, 16).astype(int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot decode {path} at frame {frame_index}")
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(np.asarray(pad(img, 172), dtype=np.float32) / 255.0)
            finally:
                cap.release()
            tensor = torch.from_numpy(np.stack(frames).transpose(3, 0, 1, 2).copy())
        return tensor, LABELS[self.task].index(row["label"])


def build_model(task, pretrained):
    if task == "smoking":
        import timm

        return timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=pretrained, num_classes=2)
    from movinets import MoViNet
    from movinets.config import _C

    model = MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=pretrained)
    old = model.classifier[-1]
    model.classifier[-1] = nn.Conv3d(old.conv_1.conv3d.in_channels, 2, kernel_size=1, bias=True)
    return model


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probabilities.extend(model(x.to(device)).softmax(dim=1)[:, 1].cpu().tolist())
            labels.extend(y.tolist())
    pred = [int(p >= threshold) for p in probabilities]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    curve_precision, curve_recall, _ = precision_recall_curve(labels, probabilities)
    return (
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": auc(curve_recall, curve_precision),
            "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
        },
        labels,
        probabilities,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=LABELS)
    p.add_argument("output", type=Path)
    p.add_argument("--allow-unreviewed", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "artifacts").resolve()) or out.exists():
        p.error("output must be a new directory inside artifacts/")
    if not args.allow_unreviewed:
        p.error("source annotations are unreviewed; pass --allow-unreviewed for an exploratory run")
    data = validate_splits(args.task)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    model = build_model(args.task, pretrained=True).to(device)
    batch = 16 if args.task == "smoking" else 2
    accumulation = 1 if args.task == "smoking" else 4
    train = DataLoader(Samples(args.task, data["train"], True), batch_size=batch, shuffle=True, num_workers=0)
    val = DataLoader(Samples(args.task, data["val"]), batch_size=batch, num_workers=0)
    counts = np.bincount([LABELS[args.task].index(row["label"]) for row in data["train"]], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(data["train"]) / (2 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    out.mkdir(parents=True)
    metadata = {
        "task": args.task,
        "labels": LABELS[args.task],
        "seed": 42,
        "status": "exploratory_source_labels_unreviewed",
        "batch_size": batch,
        "accumulation": accumulation,
        "epochs_max": args.epochs,
        "manifest_sha256": {
            s: hashlib.sha256((ROOT / "data/splits" / f"{args.task}_v1" / f"{s}.csv").read_bytes()).hexdigest() for s in data
        },
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    best, stale = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train):
            (loss_fn(model(x.to(device)), y.to(device)) / accumulation).backward()
            if (i + 1) % accumulation == 0 or i + 1 == len(train):
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in val:
                total += loss_fn(model(x.to(device)), y.to(device)).item() * len(y)
                n += len(y)
        val_loss = total / n
        print(f"{args.task} epoch {epoch + 1}: val_loss={val_loss:.5f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": best, "metadata": metadata}, out / "best.pt")
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    _, val_labels, val_probs = evaluate(model, val, device)
    thresholds = sorted(set([0.0, 1.0, *val_probs]))
    threshold = max(
        thresholds,
        key=lambda t: precision_recall_fscore_support(
            val_labels, [int(x >= t) for x in val_probs], average="binary", zero_division=0
        )[2],
    )
    metrics, _, _ = evaluate(model, DataLoader(Samples(args.task, data["test"]), batch_size=batch), device, threshold)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "threshold_source": "validation_f1",
                "test": metrics,
                "warning": "preliminary, source labels unreviewed",
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
