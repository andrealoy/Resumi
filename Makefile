.PHONY: install lint test run

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test:
	uv run pytest

run:
	uv run uvicorn resumi.main:app --reload --app-dir src
