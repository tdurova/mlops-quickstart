Constraint level: High

Scope and non-goals
- Hard constraints: no new dependencies; no API response shape changes unless tests are updated first; use the Makefile + uv workflow only; Postgres in compose.yaml is a placeholder—no database usage; no background jobs/queues; no heavy ML stacks (pandas/torch/tensorflow).
- Runtime facts: Python 3.11 FastAPI app in src/app.py; packaging via uv with pyproject.toml/uv.lock; Make targets: setup, dev, test, lint, fmt, typecheck, up, down, logs, clean.
- Ports: APP_PORT (default 8080) is the only port env var; used by make dev and docker compose (`${APP_PORT:-8080}` mapping).
- Env: env.example defines APP_PORT plus optional Postgres vars (unused by the app).

How to run
- Dev server: `make dev` (honors APP_PORT, default 8080).
- Tests: `make test`
- Lint/format/typecheck: `make lint`, `make fmt`, `make typecheck`
- Docker: `make up` / `make down`, logs with `make logs`

Lifecycle and readiness
- Model lifecycle (current, test-backed): src/model.py trains an Iris StandardScaler + LogisticRegression pipeline during FastAPI lifespan startup; stored on app.state.model. Module-level mutable globals are forbidden.
- Readiness — current behavior (tests-backed): /health returns {"status": "ok"} with 200 only when the model is ready, else 503; /predict returns 503 {"detail": "Model not loaded"} if missing. Implemented via centralized dependency gate (get_model).

API contracts (grounded in tests/src)
- Request schema: POST /predict with JSON body {"values": [f1, f2, f3, f4]} where values is exactly four numbers; ints are coerced to float.
- Success response: 200 with JSON {"prediction": <int>, "probabilities": <list[float]>}. Probabilities are length 3 (Iris classes) and sum to ~1.0. Example: {"prediction": 0, "probabilities": [0.9, 0.05, 0.05]}.
- Validation errors: status 400 with {"detail": "Invalid request", "errors": [...]}; this comes from the RequestValidationError handler in src/app.py and is enforced by tests.
- Readiness failure: when the model is absent, /predict must return 503 with the same stable JSON shape used today.

Dependency and performance policy
- Heavy deps: avoid adding new heavy dependencies (e.g., torch/tensorflow). pandas was removed as unused; tighten to a full prohibition on heavy ML stacks. Do not touch uv.lock unless intentionally changing deps.
- CI cadence (see .github/workflows/test.yml): uv sync --frozen --group dev, uv run pytest -q, uv run ruff check ., uv run mypy .
- Local budget: keep tests under ~5 seconds on a dev machine; avoid adding slow startup hooks or per-request work.

Docker rule
- Builds must use a .dockerignore that excludes .venv, caches (.pytest_cache, .mypy_cache, .ruff_cache), __pycache__, .git, and artifacts.

Conflict and drift policy
- If tests, code, and this guide disagree, treat it as a bug and reconcile in order: tests (contract) → code → guide. Do not treat “tests > code > docs” as permission to drift; fix the inconsistency promptly.

