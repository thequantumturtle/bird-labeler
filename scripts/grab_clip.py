from __future__ import annotations

from pathlib import Path
import subprocess

from yt_dlp import YoutubeDL


def main() -> None:
    repo = Path(r"c:\Users\djust\Projects\bird-labeler")
    clips = repo / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    url = "https://www.youtube.com/watch?v=_2_nRhYWtlY"
    start = "00:10:00"
    dur = "00:00:10"
    name = "yt_2_nRhYWtlY_10m00s_10s.mp4"

    tmp = clips / f"_tmp_{name}"
    out = clips / name
    if tmp.exists():
        tmp.unlink()
    if out.exists():
        out.unlink()

    ydl_opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "outtmpl": str(tmp),
        "download_sections": [f"*{start}-{dur}"],
        "quiet": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            start,
            "-i",
            str(tmp),
            "-t",
            dur,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-shortest",
            "-c",
            "copy",
            str(out),
        ],
        check=True,
    )

    tmp.unlink(missing_ok=True)
    print(out)


if __name__ == "__main__":
    main()
