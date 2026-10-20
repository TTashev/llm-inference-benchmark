"""Lightweight per-process observability without external monitoring services.

The request middleware records HTTP outcomes, generates correlation IDs, and
writes JSON log events without logging user prompts or generated text.
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from starlette.responses import JSONResponse


logger = logging.getLogger("inference_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def log_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    """Emit one JSON object per event; do not include request or model text."""
    logger.log(
        level,
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **fields,
            },
            separators=(",", ":"),
        ),
    )


class ServiceMetrics:
    """Thread-safe, process-local counters and aggregate durations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at = time.perf_counter()
        self._generation_requests = 0
        self._generation_success = 0
        self._generation_rejected = 0
        self._generation_client_errors = 0
        self._generation_server_errors = 0
        self._generation_other = 0
        self._generation_in_flight = 0
        self._http_duration_total_ms = 0.0
        self._inference_completed = 0
        self._inference_duration_total_ms = 0.0
        self._lock_wait_total_ms = 0.0

    def admitted(self) -> None:
        with self._lock:
            self._generation_in_flight += 1

    def released(self) -> None:
        with self._lock:
            self._generation_in_flight -= 1

    def record_inference(self, inference_ms: float, lock_wait_ms: float) -> None:
        with self._lock:
            self._inference_completed += 1
            self._inference_duration_total_ms += inference_ms
            self._lock_wait_total_ms += lock_wait_ms

    def record_http_response(self, status_code: int, duration_ms: float) -> None:
        """Count completed /generate responses only; do not count /metrics polls."""
        with self._lock:
            self._generation_requests += 1
            self._http_duration_total_ms += duration_ms
            if 200 <= status_code < 300:
                self._generation_success += 1
            elif status_code == 503:
                self._generation_rejected += 1
            elif 400 <= status_code < 500:
                self._generation_client_errors += 1
            elif 500 <= status_code < 600:
                self._generation_server_errors += 1
            else:
                self._generation_other += 1

    def snapshot(self) -> dict:
        """Take a consistent snapshot with one short lock hold."""
        with self._lock:
            requests = self._generation_requests
            inferences = self._inference_completed
            return {
                "uptime_s": round(time.perf_counter() - self._started_at, 2),
                "generation_requests_total": requests,
                "generation_success_total": self._generation_success,
                "generation_rejected_503_total": self._generation_rejected,
                "generation_client_errors_total": self._generation_client_errors,
                "generation_server_errors_total": self._generation_server_errors,
                "generation_other_total": self._generation_other,
                "generation_in_flight": self._generation_in_flight,
                "inference_completed_total": inferences,
                "avg_generation_http_latency_ms": (
                    round(self._http_duration_total_ms / requests, 2)
                    if requests else None
                ),
                "avg_inference_ms": (
                    round(self._inference_duration_total_ms / inferences, 2)
                    if inferences else None
                ),
                "avg_lock_wait_ms": (
                    round(self._lock_wait_total_ms / inferences, 2)
                    if inferences else None
                ),
            }


def request_logging_enabled() -> bool:
    return os.getenv("LOG_REQUESTS", "1").strip().lower() not in (
        "0", "false", "off", "no"
    )


class RequestObservabilityMiddleware:
    """Pure ASGI middleware: request IDs, failure-safe responses and JSON logs."""

    def __init__(self, app, metrics: ServiceMetrics, *, log_requests: bool = True):
        self.app = app
        self.metrics = metrics
        self.log_requests = log_requests

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        request_id_header = (b"x-request-id", request_id.encode("ascii"))
        path = scope.get("path", "")
        method = scope.get("method", "")
        start = time.perf_counter()
        response_started = False
        status_code = 500

        async def send_with_request_id(message):
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                # Replace an upstream X-Request-ID rather than emitting duplicates.
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                message["headers"] = headers + [request_id_header]
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            # Never expose exception messages (which could contain user prompts).
            log_event(
                "request_exception",
                level=logging.ERROR,
                request_id=request_id,
                method=method,
                path=path,
                error_type=type(exc).__name__,
            )
            if response_started:
                # The HTTP status/body is already on the wire; cannot replace it.
                raise
            response = JSONResponse(
                {"detail": "Internal server error"},
                status_code=500,
            )
            await response(scope, receive, send_with_request_id)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            if path == "/generate":
                self.metrics.record_http_response(status_code, duration_ms)
            if self.log_requests:
                log_event(
                    "http_request",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=round(duration_ms, 2),
                )
