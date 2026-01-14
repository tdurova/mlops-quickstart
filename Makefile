APP_PORT ?= 8080

.PHONY: setup dev test lint fmt typecheck up down logs clean

# One-command setup: checks for uv, then syncs deps (including dev)
setup:
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv is required but not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync --group dev

# Local development server (no activation needed)
dev:
	uv run uvicorn src.app:app --reload --host 0.0.0.0 --port $(APP_PORT)

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy .

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
