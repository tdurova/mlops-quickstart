# mlops-quickstart

An opinionated, production-shaped MLOps starter: a Dockerized FastAPI inference service (sklearn) with CI and a Cloud Run deploy workflow.

## Why this exists
- Show end-to-end “ship a model as a service” basics (validation, readiness, logging, Docker, CI/CD) without heavyweight infra.
- Be small enough to understand in one sitting, but structured enough to look like real work in a 2026 MLOps interview.

## Architecture at a glance
- **API**: FastAPI (`src/app.py`) with `/health` (readiness-gated) and `/predict`.
- **Model lifecycle**: trains an Iris `StandardScaler + LogisticRegression` pipeline on startup and stores it on `app.state.model` (`src/model.py`).
- **Contracts**: request/response and validation behavior are test-backed (`tests/test_app.py`).
- **Observability**: JSON logs + request-id propagation middleware (`src/logging_util.py`).
- **Packaging**: `uv` + `pyproject.toml`/`uv.lock`; common tasks via `Makefile`.
- **Container**: `Dockerfile` runs `uvicorn` and honors Cloud Run’s `PORT`.

## Prerequisites
- Docker
- uv (modern Python package manager)

## Quickstart (5 minutes)

1. Install uv (once, if not installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Setup project (automatically handles Python version + dependencies):
   ```bash
   make setup
   ```

3. Run locally:
   ```bash
   make dev
   ```

4. Or start the full stack with Docker:
   ```bash
   # Optional: copy and edit `.env` if you want to override defaults
   cp .env.example .env
   make up
   ```

5. Test the API:
   ```bash
   curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"values": [5.1,3.5,1.4,0.2]}'
   ```

## Make targets
- `make setup` – install project + dev deps with uv
- `make dev` – run the FastAPI app with reload
- `make test` – run pytest
- `make lint` / `make fmt` – ruff check/format
- `make typecheck` – mypy
- `make up` / `make down` – docker compose up/down
- `make logs` – tail the API logs

## Configuration
`.env` is optional. Start from `.env.example` (or `env.example`) and override as needed.

## Services (docker compose)
- `api`: builds from `Dockerfile` and serves HTTP on container port `8080` (mapped to host `${APP_PORT:-8080}`).
- `db`: optional Postgres 16 placeholder for future examples (the current demo app does not use a database).

## Logging
- Structured logs are NDJSON to stdout (Datadog-friendly), tagged with `service`, `env`, `version`, and `request_id`.
- Incoming `X-Request-ID` is honored (or generated); responses echo it, and each request log includes path/method/status/duration_ms.
- Control verbosity with `LOG_LEVEL` (default INFO); 4xx log at WARN, 5xx at ERROR with stack traces; no raw payloads are logged by default.

## Deploy (Cloud Run)
This repo ships with a GitHub Actions workflow that builds a container (Cloud Build), pushes it to Artifact Registry, and deploys to Cloud Run after CI succeeds on `main`: `.github/workflows/deploy-cloudrun.yml`.

Required GitHub secrets:
- `GCP_PROJECT_ID`
- `GCP_REGION` (example: `us-central1`)
- `GCP_ARTIFACT_REPO` (Artifact Registry Docker repo name)
- `CLOUD_RUN_SERVICE` (Cloud Run service name)
- `GCP_WIF_PROVIDER` and `GCP_WIF_SERVICE_ACCOUNT` (recommended: Workload Identity Federation; no long-lived keys)

## Engineering guidelines (small, practical)
- KISS: keep modules small and focused; avoid magic config when defaults suffice.
- Clean code: prefer clear names, short functions, and single-purpose modules.
- DDD-lite: treat `src/` as the domain boundary; keep infra concerns (I/O, Docker, CI) at the edges.
- TDD: add/adjust tests in `tests/` alongside behavior changes; keep tests fast and deterministic.

## CI
`.github/workflows/test.yml` runs: `uv sync --frozen --group dev` → `uv run pytest -q` → `uv run ruff check .` → `uv run mypy .`.

## Ops
- **Health/readiness**: `GET /health` returns `200 {"status":"ok"}` only when the model is loaded; otherwise `503`.
- **Logs**: `make logs` (Docker) or Cloud Run Logs Explorer; correlate requests via `X-Request-ID`.

## Trade-offs
- **Why Cloud Run**: minimal ops surface area, fast iteration, and “production enough” for an inference microservice.
- **Why not Kubernetes**: this repo is intentionally small; adding GKE/Helm would add complexity without a demonstrated requirement.
