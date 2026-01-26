import asyncio
import json
import logging
import os
from io import StringIO
from contextlib import suppress
from uuid import UUID
import pytest
import httpx
from src.app import app, get_model
from src.logging_util import JSONFormatter, _sanitize_request_id


@pytest.fixture()
def api_request():
    def _request(
        method: str,
        path: str,
        json: dict | None = None,
        headers: dict[str, str] | None = None,
    ):
        async def _call():
            # Keep an additional task alive to avoid event-loop edge cases in this environment.
            sleeper = asyncio.create_task(asyncio.sleep(0.1))
            transport = httpx.ASGITransport(app=app)
            try:
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(
                        transport=transport, base_url="http://test"
                    ) as c:
                        return await c.request(method, path, json=json, headers=headers)
            finally:
                sleeper.cancel()
                with suppress(asyncio.CancelledError):
                    await sleeper

        return asyncio.run(_call())

    return _request


def test_health(api_request):
    r = api_request("GET", "/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict(api_request):
    # Iris sample (first row)
    sample = [5.1, 3.5, 1.4, 0.2]
    r = api_request("POST", "/predict", json={"values": sample})
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert isinstance(data["probabilities"], list)


def test_predict_probabilities_shape_and_sum(api_request):
    sample = [5.1, 3.5, 1.4, 0.2]
    r = api_request("POST", "/predict", json={"values": sample})
    assert r.status_code == 200
    data = r.json()
    probs = data["probabilities"]
    assert isinstance(probs, list)
    assert len(probs) == 3
    # Sum of probabilities should be ~1.0 (allow small numerical error).
    assert abs(sum(probs) - 1.0) < 1e-6


def test_predict_wrong_length_returns_400(api_request):
    r = api_request("POST", "/predict", json={"values": [1.0, 2.0, 3.0]})
    assert r.status_code == 400
    data = r.json()
    assert data["detail"] == "Invalid request"
    assert data["errors"]


def test_predict_non_numeric_returns_400(api_request):
    r = api_request("POST", "/predict", json={"values": [1.0, "bad", 3.0, 4.0]})
    assert r.status_code == 400
    data = r.json()
    assert data["detail"] == "Invalid request"


def test_predict_null_in_payload_returns_400(api_request):
    r = api_request("POST", "/predict", json={"values": [1.0, None, 3.0, 4.0]})
    assert r.status_code == 400
    data = r.json()
    assert data["detail"] == "Invalid request"


def test_predict_too_many_values_returns_400(api_request):
    r = api_request("POST", "/predict", json={"values": [1, 2, 3, 4, 5]})
    assert r.status_code == 400
    data = r.json()
    assert data["detail"] == "Invalid request"
    assert data["errors"]


def test_predict_missing_values_field_returns_400(api_request):
    r = api_request("POST", "/predict", json={})
    assert r.status_code == 400
    data = r.json()
    assert data["detail"] == "Invalid request"
    assert data["errors"]


def test_predict_accepts_ints_via_coercion(api_request):
    r = api_request("POST", "/predict", json={"values": [5, 3, 1, 0]})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probabilities"], list)


def test_request_id_propagation_and_json_logs(api_request):
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    formatter = JSONFormatter(
        os.getenv("SERVICE_NAME", "mlops-quickstart"),
        os.getenv("APP_ENV", "local"),
        os.getenv("APP_VERSION", "dev"),
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        request_id = "req-demo-123"
        r = api_request("GET", "/health", headers={"X-Request-ID": request_id})
        assert r.status_code == 200
        assert r.headers.get("X-Request-ID") == request_id

        handler.flush()
        captured = buffer.getvalue().strip().splitlines()
        assert captured, "expected logs to be emitted"

        parsed = [json.loads(line) for line in captured]
        assert all(isinstance(entry, dict) for entry in parsed)
        assert any(entry.get("request_id") == request_id for entry in parsed)
        assert all("timestamp" in entry for entry in parsed)
    finally:
        root_logger.removeHandler(handler)


def test_request_id_sanitization_rejects_control_chars():
    raw = "req-demo-123\r\nX-Evil: 1"
    sanitized = _sanitize_request_id(raw)
    assert sanitized != raw.strip()
    UUID(sanitized)


def test_request_id_differs_per_request(api_request):
    first = api_request("GET", "/health")
    second = api_request("GET", "/health")

    assert first.status_code == 200
    assert second.status_code == 200

    first_id = first.headers.get("X-Request-ID")
    second_id = second.headers.get("X-Request-ID")

    assert first_id
    assert second_id
    assert first_id != second_id


def test_middleware_logs_4xx_at_warning_level(api_request):
    """Verify 4xx responses are logged at WARNING level."""
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    formatter = JSONFormatter(
        os.getenv("SERVICE_NAME", "mlops-quickstart"),
        os.getenv("APP_ENV", "local"),
        os.getenv("APP_VERSION", "dev"),
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    logger = logging.getLogger("app.request")
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    try:
        r = api_request("POST", "/predict", json={"values": [1.0, 2.0]})
        assert r.status_code == 400

        handler.flush()
        captured = buffer.getvalue().strip().splitlines()
        assert captured, "expected logs to be emitted"

        parsed = [json.loads(line) for line in captured]
        request_logs = [e for e in parsed if e.get("event") == "request"]
        assert request_logs, "expected request log entry"
        assert any(e.get("level") == "warning" for e in request_logs)
        assert any(e.get("status") == 400 for e in request_logs)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def test_middleware_logs_5xx_at_error_level(api_request, monkeypatch):
    """Ensure 5xx responses emit error-level logs and include request_id."""
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    formatter = JSONFormatter(
        os.getenv("SERVICE_NAME", "mlops-quickstart"),
        os.getenv("APP_ENV", "local"),
        os.getenv("APP_VERSION", "dev"),
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("app.request")
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    class BadModel:
        def predict(self, _):  # pragma: no cover - simple stub
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail="boom")

    def bad_model_dep():
        return BadModel()

    monkeypatch.setitem(app.dependency_overrides, get_model, bad_model_dep)

    try:
        r = api_request("POST", "/predict", json={"values": [5, 3, 1, 0]})
        assert r.status_code == 500

        handler.flush()
        captured = buffer.getvalue().strip().splitlines()
        parsed = [json.loads(line) for line in captured]
        request_logs = [e for e in parsed if e.get("event") == "request"]
        assert request_logs, "expected request log entry"
        assert any(e.get("level") == "error" for e in request_logs)
        assert any(e.get("request_id") for e in request_logs)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        app.dependency_overrides.pop(get_model, None)


def test_health_returns_503_when_model_missing(api_request, monkeypatch):
    def missing_model():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model not loaded")

    monkeypatch.setitem(app.dependency_overrides, get_model, missing_model)
    try:
        r = api_request("GET", "/health")
        assert r.status_code == 503
        assert r.json() == {"detail": "Model not loaded"}
    finally:
        app.dependency_overrides.pop(get_model, None)


def test_predict_returns_503_when_model_missing(api_request, monkeypatch):
    def missing_model():
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model not loaded")

    monkeypatch.setitem(app.dependency_overrides, get_model, missing_model)
    try:
        sample = [5.1, 3.5, 1.4, 0.2]
        r = api_request("POST", "/predict", json={"values": sample})
        assert r.status_code == 503
        assert r.json() == {"detail": "Model not loaded"}
    finally:
        app.dependency_overrides.pop(get_model, None)


def test_middleware_logs_request_metadata(api_request):
    """Verify request logs include path, method, status, and duration."""
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    formatter = JSONFormatter(
        os.getenv("SERVICE_NAME", "mlops-quickstart"),
        os.getenv("APP_ENV", "local"),
        os.getenv("APP_VERSION", "dev"),
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("app.request")
    logger.addHandler(handler)

    try:
        r = api_request("GET", "/health")
        assert r.status_code == 200

        handler.flush()
        captured = buffer.getvalue().strip().splitlines()
        parsed = [json.loads(line) for line in captured]
        request_logs = [e for e in parsed if e.get("event") == "request"]

        assert request_logs, "expected request log entry"
        entry = request_logs[0]
        assert entry.get("path") == "/health"
        assert entry.get("method") == "GET"
        assert entry.get("status") == 200
        assert "duration_ms" in entry
        assert isinstance(entry["duration_ms"], (int, float))
    finally:
        logger.removeHandler(handler)
