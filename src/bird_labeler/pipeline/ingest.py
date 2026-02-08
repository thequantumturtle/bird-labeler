from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def normalize_av(input_path: Path, output_path: Path) -> Path:
    """Normalize audio/video duration by remuxing with -shortest."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot normalize audio/video streams")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_copy = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd_copy, capture_output=True, text=True)
    if result.returncode == 0:
        return output_path

    cmd_reencode = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd_reencode, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg normalization failed"
        raise RuntimeError(stderr)
    return output_path
