from __future__ import annotations

import json
import logging

from src.logging_util import (
    JSONFormatter,
    _log_level,
    _sanitize_request_id,
    configure_logging,
)


def test_log_level_accepts_valid_names(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert _log_level() == "DEBUG"


def test_log_level_falls_back_to_info_for_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "not-a-level")
    assert _log_level() == "INFO"


def test_sanitize_request_id_rejects_empty_and_too_long() -> None:
    assert _sanitize_request_id("") != ""
    long_id = "a" * 200
    sanitized = _sanitize_request_id(long_id)
    assert sanitized != long_id
    assert len(sanitized) <= 128


def test_configure_logging_applies_env_fields(monkeypatch, capsys) -> None:
    monkeypatch.setenv("SERVICE_NAME", "demo-service")
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("APP_VERSION", "1.2.3")
    monkeypatch.setenv("LOG_LEVEL", "warning")

    configure_logging()
    logger = logging.getLogger("uvicorn")

    logger.warning("hello", extra={"event": "ping"})

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "expected log output"
    parsed = json.loads(captured[-1])
    assert parsed["service"] == "demo-service"
    assert parsed["env"] == "staging"
    assert parsed["version"] == "1.2.3"
    assert parsed["level"] == "warning"


def test_json_formatter_includes_service_env_version(monkeypatch) -> None:
    formatter = JSONFormatter("svc", "prod", "9.9.9")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    output = json.loads(formatter.format(record))
    assert output["service"] == "svc"
    assert output["env"] == "prod"
    assert output["version"] == "9.9.9"
    assert output["message"] == "hello"
    assert "timestamp" in output
