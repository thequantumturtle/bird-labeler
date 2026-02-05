from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from bird_labeler.cli import app, run_pipeline


def _write_test_video(path: Path, frame_count: int = 10) -> None:
    fps = 10
    width, height = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    rng = np.random.default_rng(123)

    for _ in range(frame_count):
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def test_smoke_run(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_test_video(input_path, frame_count=10)

    run_pipeline(
        input=input_path,
        out=output_path,
        config=Path("configs/default.yaml"),
        verbose=False,
    )

    assert output_path.exists()
    assert _count_frames(output_path) == 10


def test_smoke_run_with_fakes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output_fakes.mp4"
    _write_test_video(input_path, frame_count=10)

    run_pipeline(
        input=input_path,
        out=output_path,
        config=Path("configs/default.yaml"),
        detector="fake",
        verbose=False,
    )

    assert output_path.exists()
    assert _count_frames(output_path) == 10


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"], prog_name="bird-labeler")
    assert result.exit_code == 0
    assert "Usage: bird-labeler" in result.output
