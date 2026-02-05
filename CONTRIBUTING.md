# Contributing

Thanks for helping improve Bird Labeler.

## Local checks

Run these before opening a PR:

```powershell
ruff format .
ruff check .
pytest -q
```

If you use pre-commit, install the pre-push hook:

```powershell
pre-commit install --hook-type pre-push
```

## CI

CI runs on every push and pull request:

- `ruff format --check .`
- `ruff check .`
- `pytest -q`
