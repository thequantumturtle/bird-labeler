from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bird_labeler.pipeline.detect import Detection


@dataclass(frozen=True)
class TrackedDetection:
    detection: Detection
    track_id: int


class Tracker(Protocol):
    def update(self, detections: list[Detection]) -> list[TrackedDetection]: ...


def _iou(a: Detection, b: Detection) -> float:
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _TrackState:
    track_id: int
    detection: Detection
    age: int = 0


class IouTracker:
    def __init__(self, max_age: int = 15, iou_thresh: float = 0.3) -> None:
        self._max_age = max_age
        self._iou_thresh = iou_thresh
        self._next_id = 1
        self._tracks: list[_TrackState] = []

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Compute all IOUs for greedy matching.
        pairs: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx, det in enumerate(detections):
                pairs.append((_iou(track.detection, det), t_idx, d_idx))
        pairs.sort(key=lambda p: p[0], reverse=True)

        for score, t_idx, d_idx in pairs:
            if score < self._iou_thresh:
                break
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            self._tracks[t_idx].detection = detections[d_idx]
            self._tracks[t_idx].age = 0

        # Age unmatched tracks.
        for idx, track in enumerate(self._tracks):
            if idx not in matched_tracks:
                track.age += 1

        # Drop old tracks.
        self._tracks = [t for t in self._tracks if t.age <= self._max_age]

        # Create new tracks for unmatched detections.
        for d_idx, det in enumerate(detections):
            if d_idx in matched_dets:
                continue
            self._tracks.append(_TrackState(track_id=self._next_id, detection=det, age=0))
            self._next_id += 1

        return [
            TrackedDetection(detection=track.detection, track_id=track.track_id)
            for track in self._tracks
        ]
