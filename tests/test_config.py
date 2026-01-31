from __future__ import annotations

from pathlib import Path

import yaml


def test_config_defaults() -> None:
    config_path = Path("configs/default.yaml")
    data = yaml.safe_load(config_path.read_text())

    assert "input" in data
    assert "output" in data
    assert "processing" in data
    assert "max_frames" in data["processing"]
