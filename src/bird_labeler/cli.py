from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import typer

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


@app.command()
def run(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Input video path"),
    out: Path = typer.Option(..., "--out", help="Output video path"),
    config: Path | None = typer.Option(
        None, "--config", exists=True, readable=True, help="Config path"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))

    max_frames = 30
    count = 0
    logger.info("Processing up to %d frames", max_frames)

    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        count += 1
        if count % 5 == 0:
            logger.info("Processed %d frames", count)

    cap.release()
    writer.release()
    logger.info("Wrote %d frames to %s", count, out)


if __name__ == "__main__":
    app()
