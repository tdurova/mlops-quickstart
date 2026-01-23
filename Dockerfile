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

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8080}/health', timeout=5)" || exit 1

CMD ["sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
