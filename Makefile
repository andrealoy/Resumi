.PHONY: install lint test agent-test run clean

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test:
	uv run pytest

agent-test:
	uv run pytest tests/test_tool_routing.py -v

run:
	uv run uvicorn resumi.main:app --reload --app-dir src

clean:
	@echo "🧹 Clearing all caches and data..."
	rm -f credentials/gmail-token.json
	rm -f local_calendar.csv
	rm -rf .data/
	rm -rf docs/
	find src -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	@echo "✅ Done. Gmail token, FAISS index, SQLite DB, documents and Python caches removed."
