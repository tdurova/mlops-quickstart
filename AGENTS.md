Constraint level: High

Scope and constraints
- Hard constraints: no new dependencies; no API response shape changes unless tests are updated first; use the Makefile + uv workflow; Postgres in compose.yaml is a placeholder—no DB usage; no background jobs/queues; avoid heavy ML stacks.
- Stack/workflow: Python 3.11 FastAPI app in src/app.py with model logic in src/model.py; packaging via uv with pyproject.toml/uv.lock; Make targets: setup, dev, test, lint, fmt, typecheck, up, down, logs, clean.
- Ports: APP_PORT (default 8080) is the only runtime port env var; used by make dev and docker compose (`${APP_PORT:-8080}` mapping).

Environment
- env.example: APP_PORT (used by app/dev/compose); POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB (only used by the compose Postgres placeholder; the FastAPI app ignores them).
- .env / .env.example: APP_ENV, LOG_LEVEL, DATABASE_URL, optional MLFLOW_TRACKING_URI and FEATURE_FLAG_EXPERIMENTAL_MODEL—currently unused by the app/tests.

Lifecycle and readiness
- Startup (tests-backed): FastAPI lifespan trains an Iris StandardScaler + LogisticRegression(max_iter=200) pipeline via train_model() and stores it on app.state.model; no module-level mutable globals (see src/model.py, src/app.py).
- Model gate (code): get_model() reads app.state.model and raises HTTPException(503, "Model not loaded") if missing; both /health and /predict depend on it.
- /health (tests-backed): returns 200 {"status": "ok"} after startup (tests/test_app.py::test_health); missing-model 503 is a code path, not covered by tests.

API behavior (tests-backed)
- Request schema: POST /predict with JSON {"values": [f1, f2, f3, f4]}; exactly four numbers, ints coerced to float (Features model + field_validator).
- Success: 200 with {"prediction": <int>, "probabilities": [p0, p1, p2]}; probabilities length 3 and sum ~1.0 (tests/test_app.py::test_predict_probabilities_shape_and_sum).
- Validation errors: 400 with {"detail": "Invalid request", "errors": [...]} from the RequestValidationError handler; triggered for wrong length, non-numeric, or None payloads (tests/test_app.py 400 cases).
- Readiness failure path (code): if the model is absent, endpoints return 503 {"detail": "Model not loaded"} via get_model(); not exercised by current tests.

Dependencies, CI, and performance
- Runtime deps: fastapi, uvicorn[standard], scikit-learn; dev deps: pytest, httpx, ruff, mypy; no pandas/heavy ML stacks present.
- CI order (.github/workflows/test.yml): uv sync --frozen --group dev → uv run pytest -q → uv run ruff check . → uv run mypy ..
- Keep startup/tests fast (~<5s); avoid slow hooks or per-request overhead.

Docker and ops
- docker compose defines api (uvicorn) and optional db Postgres placeholder; the FastAPI app does not connect to Postgres.
- .dockerignore exists; excludes .venv, .pytest_cache, .mypy_cache, .ruff_cache, __pycache__, .git, build/dist/coverage artifacts. Keep it aligned with built images.
- Commands: make up / make down; logs via make logs; clean caches with make clean.

API example
```
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"values":[5.1,3.5,1.4,0.2]}'
# -> 200 {"prediction": 0, "probabilities": [0.9847, 0.0153, 0.0000]} (values may vary slightly)
```

Conflict and drift policy
- If tests, code, and this guide disagree, treat tests as the contract, then code, then docs; record TODOs for ambiguities rather than guessing.
