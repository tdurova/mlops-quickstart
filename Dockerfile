FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies into venv
COPY pyproject.toml .python-version ./
COPY uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy code
COPY src/ ./src

# Add venv to PATH (pro move: no activation needed)
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
