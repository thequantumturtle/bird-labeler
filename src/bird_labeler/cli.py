from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import typer

from bird_labeler.pipeline.classify import FakeClassifier, HfBirdClassifier
from bird_labeler.pipeline.detect import FakeDetector, YoloBirdDetector
from bird_labeler.pipeline.track import IouTracker

app = typer.Typer(help="Bird labeling pipeline CLI.")


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


def run_pipeline(
    *,
    input: Path,
    out: Path,
    config: Path | None = None,
    detector: str = "fake",
    yolo_weights: Path | None = None,
    device: str | None = None,
    imgsz: int = 640,
    classifier: str = "off",
    classifier_model: str | None = None,
    classifier_device: str | None = None,
    tracking: str = "iou",
    max_age: int = 15,
    iou_thresh: float = 0.3,
    process_fps: float = 15.0,
    classify_every_seconds: float = 1.0,
    max_frames: int | None = None,
    verbose: bool = False,
) -> None:
    """Smoke pipeline: copy up to 30 frames from input to output."""
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
    out.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(fps / process_fps))) if process_fps > 0 else 1
    out_fps = fps / step
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
    if detector_choice == "yolo":
        weights = str(yolo_weights) if yolo_weights else "yolov8s.pt"
        device_name = device or _default_device()
        detector_impl = YoloBirdDetector(weights=weights, device=device_name, imgsz=imgsz)
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
    frame_idx = 0
    logger.info("Processing up to %d frames", max_frames)

    while max_frames < 0 or count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if step > 1 and (frame_idx % step) != 0:
            frame_idx += 1
            continue
        if detector_impl:
            detections = detector_impl.detect(frame)
            tracked = tracker.update(detections) if tracker else None
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
                    label = f"{label} #{track_id}"
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (det.x1, max(det.y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        writer.write(frame)
        count += 1
        frame_idx += 1
        if count % 5 == 0:
            logger.info("Processed %d frames", count)

    cap.release()
    writer.release()
    if chosen_codec == "mp4v":
        _reencode_h264(out, logger)
    logger.info("Wrote %d frames to %s", count, out)


@app.command()
def run(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Input video path"),
    out: Path = typer.Option(..., "--out", help="Output video path"),
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
        15.0, "--process-fps", help="Target processing FPS (drops frames)"
    ),
    imgsz: int = typer.Option(640, "--imgsz", help="YOLO inference image size"),
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
    tracking: str = typer.Option("iou", "--tracking", help="Tracking mode", case_sensitive=False),
    max_age: int = typer.Option(15, "--max-age", help="Max age for tracks in frames"),
    iou_thresh: float = typer.Option(0.3, "--iou-thresh", help="IOU threshold for tracking"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    run_pipeline(
        input=input,
        out=out,
        config=config,
        detector=detector,
        yolo_weights=yolo_weights,
        device=device,
        imgsz=imgsz,
        classifier=classifier,
        classifier_model=classifier_model,
        classifier_device=classifier_device,
        tracking=tracking,
        max_age=max_age,
        iou_thresh=iou_thresh,
        process_fps=process_fps,
        classify_every_seconds=classify_every_seconds,
        max_frames=max_frames,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
