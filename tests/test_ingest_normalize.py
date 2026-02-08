from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from bird_labeler.pipeline.ingest import normalize_av


def test_normalize_av_uses_shortest(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"fake")

    monkeypatch.setattr("shutil.which", lambda _: "ffmpeg")

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output=True, text=True):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = normalize_av(input_path, output_path)
    assert result == output_path
    assert calls
    cmd = calls[0]
    assert "-shortest" in cmd
    assert ["-map", "0:v:0"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["-map", "0:a:0"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]


def test_run_calls_normalize_av(monkeypatch, tmp_path: Path) -> None:
    import bird_labeler.cli as cli

    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"fake")

    called = {"normalize": False, "run": False}

    def fake_normalize(src: Path, dst: Path) -> Path:
        called["normalize"] = True
        return dst

    def fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        called["run"] = True
        assert kwargs["input"] != input_path
        assert kwargs["source_input"] == input_path

    monkeypatch.setattr(cli, "normalize_av_file", fake_normalize)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.run(input=input_path, out=output_path, normalize_av=True)

    assert called["normalize"]
    assert called["run"]
