from __future__ import annotations

from src.logging_util import _log_level


def test_log_level_accepts_valid_names(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert _log_level() == "DEBUG"


def test_log_level_falls_back_to_info_for_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "not-a-level")
    assert _log_level() == "INFO"

