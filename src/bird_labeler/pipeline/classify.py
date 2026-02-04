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
