.PHONY: install fmt lint test check

install:
	uv pip install -e . --dev

fmt:
	ruff format .

lint:
	ruff check .

test:
	pytest

check: fmt lint test
