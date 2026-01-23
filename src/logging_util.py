import contextvars
import datetime
import json
import logging
import logging.config
import os
import re
import time
import traceback
from typing import Any
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> str | None:
    return request_id_var.get()


def set_request_id(request_id: str) -> contextvars.Token[str | None]:
    return request_id_var.set(request_id)


def clear_request_id(token: contextvars.Token[str | None]) -> None:
    request_id_var.reset(token)


class JSONFormatter(logging.Formatter):
    """Minimal JSON formatter suitable for Datadog ingestion."""

    def __init__(self, service: str, env: str, version: str) -> None:
        super().__init__()
        self.service = service
        self.env = env
        self.version = version

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service,
            "env": self.env,
            "version": self.version,
        }

        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id

        for key in (
            "event",
            "path",
            "method",
            "status",
            "duration_ms",
            "component",
            "remote_addr",
            "user_agent",
            "model_event",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            etype, exc, tb = record.exc_info
            payload["error"] = {
                "type": etype.__name__ if etype else "UnknownError",
                "message": str(exc),
                "stack": "".join(traceback.format_exception(etype, exc, tb)).strip(),
            }

        return json.dumps(payload, separators=(",", ":"), default=str)


def _log_level() -> str:
    raw = os.getenv("LOG_LEVEL", "INFO").upper()
    return raw if raw in logging.getLevelNamesMapping() else "INFO"


def configure_logging() -> None:
    """Configure root + uvicorn loggers for JSON stdout."""
    service = os.getenv("SERVICE_NAME", "mlops-quickstart")
    env = os.getenv("APP_ENV", os.getenv("ENV", "local"))
    version = os.getenv("APP_VERSION", "dev")
    level = _log_level()

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
                "service": service,
                "env": env,
                "version": version,
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {
                "handlers": ["default"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": level,
                "propagate": False,
            },
        },
        "root": {"handlers": ["default"], "level": level},
    }

    logging.config.dictConfig(log_config)


_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sanitize_request_id(raw_id: str | None) -> str:
    """Honor caller-provided IDs if reasonable, otherwise generate one."""
    if raw_id:
        candidate = raw_id.strip()
        if 0 < len(candidate) <= 128 and _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
    return str(uuid4())


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add/propagate request_id and log request summaries."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.logger = logging.getLogger("app.request")

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = _sanitize_request_id(request.headers.get("X-Request-ID"))
        token = set_request_id(req_id)
        start = time.perf_counter()

        response: Response | None = None
        status_code = 500
        error: Exception | None = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:  # noqa: BLE001
            error = exc
            raise
        finally:
            try:
                duration_ms = (time.perf_counter() - start) * 1000
                level = logging.INFO
                if status_code >= 500 or error is not None:
                    level = logging.ERROR
                elif status_code >= 400:
                    level = logging.WARNING

                extra = {
                    "event": "request",
                    "component": "api",
                    "path": request.url.path,
                    "method": request.method,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "user_agent": request.headers.get("User-Agent"),
                }

                self.logger.log(
                    level, "request handled", extra=extra, exc_info=bool(error)
                )

                if response is not None:
                    try:
                        response.headers["X-Request-ID"] = req_id
                    except Exception:  # noqa: BLE001
                        self.logger.warning(
                            "failed to set X-Request-ID response header",
                            extra={
                                "event": "request_id_header_error",
                                "component": "api",
                                "path": request.url.path,
                                "method": request.method,
                            },
                            exc_info=True,
                        )
            finally:
                clear_request_id(token)
