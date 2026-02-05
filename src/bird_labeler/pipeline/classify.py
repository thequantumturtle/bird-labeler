from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LabelScore:
    label: str
    score: float


class Classifier(Protocol):
    def classify(self, crop) -> list[LabelScore]: ...


class FakeClassifier:
    def __init__(self) -> None:
        self._index = 0
        self._labels = ["sparrow", "robin", "finch", "crow", "owl"]

    def classify(self, crop) -> list[LabelScore]:
        idx = self._index
        self._index += 1
        label = self._labels[idx % len(self._labels)]
        score = 0.6 + (idx % 3) * 0.1
        return [LabelScore(label=label, score=score)]


class HfBirdClassifier:
    def __init__(self, model_id: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification  # type: ignore

        self._torch = torch
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageClassification.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()
        self._id2label = self._model.config.id2label or {}

    def classify(self, crop) -> list[LabelScore]:
        if crop is None or crop.size == 0:
            return []
        from PIL import Image

        image = Image.fromarray(crop[:, :, ::-1])
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[0]
            score, idx = probs.max(dim=0)
        label = self._id2label.get(int(idx.item()), str(int(idx.item())))
        label = label.replace("_", " ")
        return [LabelScore(label=label, score=float(score.item()))]
