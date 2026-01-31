# Bird Labeler

Quickstart (uv):

```powershell
# Install uv (if needed) and create a venv
uv venv

# Install the project in editable mode (include dev deps)
uv pip install -e . --dev

# Run tests
pytest
```

Local dev commands:

```powershell
just install
just fmt
just lint
just test
just check
```

If `just` is not available:

```powershell
make install
make fmt
make lint
make test
make check
```

Before pushing run: `just check` (or `make check`).

Run the CLI:

```powershell
bird-labeler --help
bird-labeler run --input path\to\input.mp4 --out path\to\output.mp4 --config configs\default.yaml
```
