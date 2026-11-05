import importlib
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("sklearn")
training = importlib.import_module("aitf_cv.tasks.train_baseline")


def test_pad_and_labels():
    img = training.pad(Image.new("RGB", (80, 40), "white"), 224)
    assert img.size == (224, 224)
    assert img.getpixel((0, 0)) == (0, 0, 0)
    assert training.LABELS["smoking"] == ["non_smoking", "smoking"]


def test_metrics_from_probabilities():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    result, _, _ = training.evaluate(Model(), [(logits, labels)], "cpu", threshold=0.5)
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["f1"] == 1.0
