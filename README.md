# mlops-quickstart

Minimal MLOps demo: a Dockerized FastAPI inference service with CI. Uses `uv` for packaging, `make` for tasks, and Docker Compose for local orchestration.

## Prerequisites
- Docker
- uv (modern Python package manager)

Install uv (once):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

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
   # Optional: copy and edit .env if you want to override defaults
   cp env.example .env
   docker compose up -d --build
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
`env.example` documents required variables. `.env` is optional; Compose defaults expose the API on `APP_PORT=8080` and configure Postgres credentials that match `compose.yaml`.

## Services (docker compose)
- `api`: builds from the local Dockerfile and runs `uvicorn src.app:app` on `${APP_PORT:-8080}` (uses defaults unless you provide `.env`).
- `db`: optional Postgres 16 service with a persistent volume (`db_data`) for future persistence/metrics examples (not used by the current demo app).

## Engineering guidelines (small, practical)
- KISS: keep modules small and focused; avoid magic config when defaults suffice.
- Clean code: prefer clear names, short functions, and single-purpose modules.
- DDD-lite: treat `src/` as the domain boundary; keep infra concerns (I/O, Docker, CI) at the edges.
- TDD: add/adjust tests in `tests/` alongside behavior changes; keep tests fast and deterministic.

## CI
`.github/workflows/test.yml` installs uv, syncs dev dependencies, and runs tests via `uv run pytest`.
