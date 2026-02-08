from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import typer

from bird_labeler.pipeline.classify import FakeClassifier, HfBirdClassifier
from bird_labeler.pipeline.detect import FakeDetector, YoloBirdDetector
from bird_labeler.pipeline.ingest import normalize_av as normalize_av_file
from bird_labeler.pipeline.track import IouTracker

app = typer.Typer(help="Bird labeling pipeline CLI.")

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


def _setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("bird_labeler")


def _resolve_default_config() -> Path:
    env_path = os.getenv("BIRD_LABELER_CONFIG")
    if env_path:
        return Path(env_path)
    workspace_path = Path("/workspace/configs/default.yaml")
    if workspace_path.exists():
        return workspace_path
    return Path("configs/default.yaml")


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _reencode_h264(path: Path, logger: logging.Logger) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; skipping H.264 re-encode")
        return
    tmp_path = path.with_suffix(".h264.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("ffmpeg re-encode failed; keeping original output")
        return
    try:
        tmp_path.replace(path)
        logger.info("Re-encoded output to H.264 for preview compatibility")
    except OSError:
        logger.warning("Failed to replace output with H.264 version")


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def _build_run_tag(parts: dict[str, str]) -> str:
    ordered_keys = [
        "detector",
        "weights",
        "imgsz",
        "conf",
        "classes",
        "tracking",
        "classifier",
        "process_fps",
    ]
    tokens = []
    for key in ordered_keys:
        value = parts.get(key)
        if not value:
            continue
        tokens.append(f"{key}-{value}")
    return "_".join(tokens)


def _write_sidecar(path: Path, payload: dict) -> None:
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _label_metrics(text: str) -> tuple[int, int, int]:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    return tw, th, baseline


def _draw_label(
    frame,
    text: str,
    x: int,
    y: int,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
) -> None:
    tw, th, baseline = _label_metrics(text)
    height, width = frame.shape[:2]
    x0 = max(0, min(x, max(0, width - (tw + 4))))
    y_min = th + baseline
    y_max = max(y_min, height - 1 - baseline)
    y0 = max(y_min, min(y, y_max))
    cv2.rectangle(
        frame,
        (x0, y0 - th - baseline),
        (x0 + tw + 4, y0 + baseline),
        bg_color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x0 + 2, y0),
        _FONT,
        _FONT_SCALE,
        text_color,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _expand_box(det, pad: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, det.x1 - pad)
    y1 = max(0, det.y1 - pad)
    x2 = min(width, det.x2 + pad)
    y2 = min(height, det.y2 + pad)
    return x1, y1, x2, y2


def run_pipeline(
    *,
    input: Path,
    source_input: Path | None = None,
    out: Path,
    config: Path | None = None,
    detector: str = "fake",
    yolo_weights: Path | None = None,
    device: str | None = None,
    imgsz: int = 640,
    yolo_conf: float = 0.25,
    yolo_classes: str = "bird",
    diagnostic_all_classes: bool = False,
    auto_tag_out: bool = False,
    classifier: str = "off",
    classifier_model: str | None = None,
    classifier_device: str | None = None,
    tracking: str = "iou",
    max_age: int = 15,
    iou_thresh: float = 0.3,
    process_fps: float = 15.0,
    classify_every_seconds: float = 1.0,
    max_frames: int | None = None,
    diagnostic_overlay: bool = False,
    verbose: bool = False,
) -> None:
    """Run detection/tracking/classification and write a labeled output video."""
    logger = _setup_logger(verbose)
    if config is None:
        config = _resolve_default_config()
    logger.info("Loading config: %s", config)

    cap = cv2.VideoCapture(str(input))
    if not cap.isOpened():
        raise typer.Exit(code=1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = max_frames if max_frames is not None else -1
    count = 0
    detector_choice = detector.lower()
    if detector_choice not in {"fake", "yolo"}:
        raise typer.BadParameter("Detector must be one of: fake, yolo")
    if device and device not in {"cpu", "cuda"}:
        raise typer.BadParameter("Device must be one of: cpu, cuda")
    tracking_choice = tracking.lower()
    if tracking_choice not in {"off", "iou"}:
        raise typer.BadParameter("Tracking must be one of: off, iou")
    weights = None
    if detector_choice == "yolo":
        weights = str(yolo_weights) if yolo_weights else "yolov8s.pt"
        device_name = device or _default_device()
        if diagnostic_all_classes:
            allowed_classes = None
        else:
            classes_input = yolo_classes.strip().lower()
            if classes_input in {"all", "*"}:
                allowed_classes = None
            elif classes_input in {"animal", "animals"}:
                allowed_classes = set(_ANIMAL_CLASSES)
            else:
                allowed_classes = {c.strip() for c in classes_input.split(",") if c.strip()}
            if not allowed_classes:
                raise typer.BadParameter("YOLO classes must be a non-empty list or 'all'")
        detector_impl = YoloBirdDetector(
            weights=weights,
            device=device_name,
            imgsz=imgsz,
            conf=yolo_conf,
            allowed_classes=allowed_classes,
        )
    else:
        detector_impl = FakeDetector()
    classifier_choice = classifier.lower()
    if classifier_choice not in {"off", "fake", "cub200", "hf"}:
        raise typer.BadParameter("Classifier must be one of: off, fake, cub200, hf")
    if classifier_choice == "off":
        classifier_impl = None
    elif classifier_choice == "fake":
        classifier_impl = FakeClassifier()
    else:
        model_id = classifier_model
        if classifier_choice == "cub200":
            model_id = model_id or "Emiel/cub-200-bird-classifier-swin"
        if not model_id:
            raise typer.BadParameter("Classifier model is required for hf classifier")
        classifier_device_name = classifier_device or device or _default_device()
        classifier_impl = HfBirdClassifier(model_id=model_id, device=classifier_device_name)
    tracker = None
    if tracking_choice == "iou":
        tracker = IouTracker(max_age=max_age, iou_thresh=iou_thresh)
    last_classify_time: dict[int, float] = {}
    last_label: dict[int, str] = {}
    if auto_tag_out:
        class_token = "all" if diagnostic_all_classes else yolo_classes.replace(",", "-")
        tag_parts = {
            "detector": detector_choice,
            "weights": Path(weights).stem if detector_choice == "yolo" and weights else "fake",
            "imgsz": str(imgsz) if detector_choice == "yolo" else "",
            "conf": f"{yolo_conf:.2f}" if detector_choice == "yolo" else "",
            "classes": class_token if detector_choice == "yolo" else "",
            "tracking": tracking_choice,
            "classifier": classifier_choice,
            "process_fps": f"{process_fps:.0f}",
        }
        run_tag = _build_run_tag(tag_parts)
        if run_tag:
            out = out.with_name(f"{out.stem}_{run_tag}{out.suffix}")

    out.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(fps / process_fps))) if process_fps > 0 else 1
    out_fps = fps
    writer = None
    chosen_codec = None
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(out), fourcc, out_fps, (width, height))
        if candidate.isOpened():
            writer = candidate
            chosen_codec = codec
            logger.info("Using video codec: %s", codec)
            break
    if writer is None:
        raise typer.Exit(code=1)

    frame_idx = 0
    processed_count = 0
    last_render: list[tuple[object, str]] = []
    last_raw: list[object] = []
    last_dropped: list[object] = []
    logger.info("Processing up to %d frames", max_frames)

    while max_frames < 0 or processed_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        process_this = step <= 1 or (frame_idx % step) == 0
        if detector_impl and process_this:
            detections = detector_impl.detect(frame)
            tracked = None
            dropped: list[object] = []
            if tracker:
                if hasattr(tracker, "update_with_dropped"):
                    tracked, dropped = tracker.update_with_dropped(detections)
                else:
                    tracked = tracker.update(detections)
            last_raw = list(detections)
            last_dropped = list(dropped)
            last_render = []
            for det in detections:
                crop = frame[det.y1 : det.y2, det.x1 : det.x2]
                track_id = None
                if tracked:
                    for item in tracked:
                        if item.detection == det:
                            track_id = item.track_id
                            break
                label = "bird"
                if classifier_impl:
                    if track_id is None or classify_every_seconds <= 0:
                        labels = classifier_impl.classify(crop)
                        label = labels[0].label if labels else "unknown"
                    else:
                        now = frame_idx / fps
                        last_time = last_classify_time.get(track_id, -1.0)
                        if now - last_time >= classify_every_seconds or track_id not in last_label:
                            labels = classifier_impl.classify(crop)
                            label = labels[0].label if labels else "unknown"
                            last_classify_time[track_id] = now
                            last_label[track_id] = label
                        else:
                            label = last_label[track_id]
                if track_id is not None:
                    label = f"{label} #{track_id} ({det.score:.2f})"
                last_render.append((det, label))
            processed_count += 1
        if diagnostic_overlay and last_raw:
            overlay = frame.copy()
            for det in last_raw:
                x1, y1, x2, y2 = _expand_box(det, 10, width, height)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
            for det in last_dropped:
                cv2.rectangle(overlay, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 255), 20)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            for det in last_dropped:
                text = f"{det.score:.2f}"
                tw, _, _ = _label_metrics(text)
                right_x = det.x2 - tw - 6
                _draw_label(
                    frame,
                    text,
                    right_x,
                    det.y1 + 10,
                    (0, 0, 0),
                    (0, 255, 255),
                )
            for det in last_raw:
                _draw_label(
                    frame,
                    f"{det.class_name} {det.score:.2f}",
                    det.x1 + 6,
                    det.y1,
                    (255, 255, 255),
                    (0, 0, 255),
                )
        for det, label in last_render:
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            _draw_label(
                frame,
                label,
                det.x1,
                det.y2 + 12,
                (255, 255, 255),
                (0, 128, 0),
            )
        writer.write(frame)
        count += 1
        frame_idx += 1
        if processed_count > 0 and processed_count % 5 == 0 and process_this:
            logger.info("Processed %d frames", processed_count)

    cap.release()
    writer.release()
    if chosen_codec == "mp4v":
        _reencode_h264(out, logger)
    logger.info("Wrote %d frames to %s", count, out)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "input": str(source_input or input),
        "output": str(out),
        "config": str(config),
        "detector": detector_choice,
        "yolo_weights": str(weights) if detector_choice == "yolo" and weights else None,
        "yolo_conf": yolo_conf if detector_choice == "yolo" else None,
        "yolo_classes": "all" if diagnostic_all_classes else yolo_classes,
        "imgsz": imgsz if detector_choice == "yolo" else None,
        "device": device or _default_device(),
        "classifier": classifier_choice,
        "classifier_model": classifier_model,
        "classifier_device": classifier_device,
        "tracking": tracking_choice,
        "max_age": max_age,
        "iou_thresh": iou_thresh,
        "process_fps": process_fps,
        "classify_every_seconds": classify_every_seconds,
        "max_frames": max_frames,
        "diagnostic_overlay": diagnostic_overlay,
        "diagnostic_all_classes": diagnostic_all_classes,
        "frame_count": count,
        "processed_frame_count": processed_count,
        "output_fps": out_fps,
        "source_fps": fps,
        "output_codec": chosen_codec,
    }
    _write_sidecar(out, metadata)


@app.command()
def run(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Input video path"),
    out: Path = typer.Option(..., "--out", help="Output video path"),
    normalize_av: bool = typer.Option(
        True,
        "--normalize-av/--no-normalize-av",
        help="Normalize audio/video duration before processing",
    ),
    config: Path | None = typer.Option(
        None, "--config", exists=True, readable=True, help="Config path"
    ),
    detector: str = typer.Option(
        "fake", "--detector", help="Detector to use", case_sensitive=False
    ),
    yolo_weights: Path | None = typer.Option(
        None, "--yolo-weights", help="Optional YOLO weights path"
    ),
    device: str | None = typer.Option(None, "--device", help="Device for YOLO (cpu or cuda)"),
    process_fps: float = typer.Option(
        15.0,
        "--process-fps",
        help="Target processing FPS (drops frames). Use 0 to process every frame.",
    ),
    imgsz: int = typer.Option(640, "--imgsz", help="YOLO inference image size"),
    yolo_conf: float = typer.Option(
        0.25, "--yolo-conf", help="YOLO confidence threshold (lower = more detections)"
    ),
    yolo_classes: str = typer.Option(
        "bird",
        "--yolo-classes",
        help="Allowed YOLO classes (comma list, or 'animals', or 'all')",
    ),
    diagnostic_all_classes: bool = typer.Option(
        False,
        "--diagnostic-all-classes",
        help="Override class filter and show all YOLO classes in diagnostic overlay",
    ),
    auto_tag_out: bool = typer.Option(
        False,
        "--auto-tag-out",
        help="Append a short parameter tag to the output filename",
    ),
    classify_every_seconds: float = typer.Option(
        1.0, "--classify-every-seconds", help="Per-track classification interval"
    ),
    classifier: str = typer.Option(
        "off", "--classifier", help="Classifier to use", case_sensitive=False
    ),
    classifier_model: str | None = typer.Option(
        None, "--classifier-model", help="Hugging Face model id for classifier"
    ),
    classifier_device: str | None = typer.Option(
        None, "--classifier-device", help="Device for classifier (cpu or cuda)"
    ),
    max_frames: int | None = typer.Option(
        None, "--max-frames", help="Maximum number of processed frames"
    ),
    diagnostic_overlay: bool = typer.Option(
        False, "--diagnostic-overlay", help="Show raw/tracked/dropped boxes"
    ),
    tracking: str = typer.Option("iou", "--tracking", help="Tracking mode", case_sensitive=False),
    max_age: int = typer.Option(15, "--max-age", help="Max age for tracks in frames"),
    iou_thresh: float = typer.Option(0.3, "--iou-thresh", help="IOU threshold for tracking"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    if normalize_av:
        with tempfile.TemporaryDirectory() as tmpdir:
            normalized = Path(tmpdir) / f"{input.stem}_normalized{input.suffix}"
            normalize_av_file(input, normalized)
            run_pipeline(
                input=normalized,
                source_input=input,
                out=out,
                config=config,
                detector=detector,
                yolo_weights=yolo_weights,
                device=device,
                imgsz=imgsz,
                yolo_conf=yolo_conf,
                yolo_classes=yolo_classes,
                diagnostic_all_classes=diagnostic_all_classes,
                auto_tag_out=auto_tag_out,
                classifier=classifier,
                classifier_model=classifier_model,
                classifier_device=classifier_device,
                tracking=tracking,
                max_age=max_age,
                iou_thresh=iou_thresh,
                process_fps=process_fps,
                classify_every_seconds=classify_every_seconds,
                max_frames=max_frames,
                diagnostic_overlay=diagnostic_overlay,
                verbose=verbose,
            )
    else:
        run_pipeline(
            input=input,
            source_input=input,
            out=out,
            config=config,
            detector=detector,
            yolo_weights=yolo_weights,
            device=device,
            imgsz=imgsz,
            yolo_conf=yolo_conf,
            yolo_classes=yolo_classes,
            diagnostic_all_classes=diagnostic_all_classes,
            auto_tag_out=auto_tag_out,
            classifier=classifier,
            classifier_model=classifier_model,
            classifier_device=classifier_device,
            tracking=tracking,
            max_age=max_age,
            iou_thresh=iou_thresh,
            process_fps=process_fps,
            classify_every_seconds=classify_every_seconds,
            max_frames=max_frames,
            diagnostic_overlay=diagnostic_overlay,
            verbose=verbose,
        )


if __name__ == "__main__":
    app()
