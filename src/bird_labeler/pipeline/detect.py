from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class Detector(Protocol):
    def detect(self, frame) -> list[Detection]:
        ...


class FakeDetector:
    def __init__(self) -> None:
        self._index = 0

    def detect(self, frame) -> list[Detection]:
        height, width = frame.shape[:2]
        idx = self._index
        self._index += 1
        count = (idx * 1103515245 + 12345) & 0xFFFFFFFF
        count = count % 3

        detections: list[Detection] = []
        for i in range(count):
            x1 = int((i + 1) * width / (count + 1) - width * 0.1)
            y1 = int((i + 1) * height / (count + 1) - height * 0.1)
            x2 = int((i + 1) * width / (count + 1) + width * 0.1)
            y2 = int((i + 1) * height / (count + 1) + height * 0.1)
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=0.5))
        return detections
