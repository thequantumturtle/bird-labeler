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

Run the CLI:

```powershell
bird-labeler --help
bird-labeler run --input path\to\input.mp4 --out path\to\output.mp4 --config configs\default.yaml
```
