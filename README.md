# Bird Labeler

![CI](https://github.com/thequantumturtle/bird-labeler/actions/workflows/ci.yml/badge.svg)

Quickstart (uv):

```powershell
# Install uv (if needed) and create a venv
uv venv

# Install the project in editable mode (include dev deps)
uv pip install -e . --group dev

# Run tests
pytest
```

Docker (Windows + GPU):

- Requires Docker Desktop with WSL2 and NVIDIA Container Toolkit (GPU enabled).
- Build and run:

```powershell
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm dev
```

Container defaults:

- Default config resolves to `/workspace/configs/default.yaml` if present.
- Override with `BIRD_LABELER_CONFIG=/path/to/config.yaml`.

Pre-push formatting:

- Install hooks: `pre-commit install --hook-type pre-push`
- This runs `ruff format` before `git push`.

Run the CLI:

```powershell
bird-labeler --help
bird-labeler run --input path\to\input.mp4 --out path\to\output.mp4 --config configs\default.yaml
```

Recommended near real-time settings (RTX 3080 Ti):

- `--detector yolo --device cuda --imgsz 640 --process-fps 15 --tracking iou --max-age 15 --iou-thresh 0.3 --classify-every-seconds 1.0`
