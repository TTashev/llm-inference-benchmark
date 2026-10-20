"""ASGI-level observability checks without downloading the model or httpx."""

import json
import re
import threading
import unittest
from unittest.mock import patch

import main
from observability import ServiceMetrics


class WorkingGenerator:
    def __call__(self, prompts, max_new_tokens):
        return ["fake completion"] * len(prompts)


class BrokenGenerator:
    def __call__(self, prompts, max_new_tokens):
        raise RuntimeError("secret user prompt: please do not expose me")


async def asgi_request(path: str, method: str = "GET", json_body: dict | None = None):
    """Exercise the real FastAPI ASGI stack, including its middleware."""
    body = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 10000),
        "server": ("127.0.0.1", 8000),
    }
    received = False
    messages = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await main.app(scope, receive, send)
    response_start = next(m for m in messages if m["type"] == "http.response.start")
    response_body = b"".join(
        m.get("body", b"") for m in messages if m["type"] == "http.response.body"
    )
    headers = {k.lower(): v for k, v in response_start["headers"]}
    return response_start["status"], headers, json.loads(response_body)


class ServiceObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_is_live_even_when_model_not_ready(self):
        with patch.object(main, "generator", None):
            status, headers, body = await asgi_request("/health")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")
            self.assertRegex(headers[b"x-request-id"].decode(), r"^[0-9a-f]{32}$")

            status, _, body = await asgi_request("/ready")
            self.assertEqual(status, 503)
            self.assertEqual(body["detail"], "Model not ready")

        with patch.object(main, "generator", WorkingGenerator()):
            status, _, body = await asgi_request("/ready")
            self.assertEqual(status, 200)
            self.assertEqual(body, {"status": "ready"})

    async def test_success_increments_metrics_and_returns_request_id(self):
        before = main.service_metrics.snapshot()
        with patch.object(main, "generator", WorkingGenerator()):
            with self.assertLogs("inference_service", level="INFO") as captured:
                status, headers, body = await asgi_request(
                    "/generate",
                    "POST",
                    {"prompt": "private customer text", "max_new_tokens": 1},
                )

        self.assertEqual(status, 200)
        self.assertEqual(body["generated_text"], "fake completion")
        request_id = headers[b"x-request-id"].decode()
        self.assertTrue(re.fullmatch(r"[0-9a-f]{32}", request_id))
        events = [json.loads(line.split(":", 2)[-1]) for line in captured.output]
        event = next(e for e in events if e.get("event") == "http_request")
        self.assertEqual(event["request_id"], request_id)
        self.assertEqual(event["path"], "/generate")
        self.assertEqual(event["status_code"], 200)
        self.assertNotIn("private customer text", " ".join(captured.output))

        status, headers, snapshot = await asgi_request("/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(snapshot["max_in_flight"], main.MAX_IN_FLIGHT)
        self.assertEqual(
            snapshot["generation_success_total"], before["generation_success_total"] + 1
        )
        self.assertEqual(
            snapshot["inference_completed_total"],
            before["inference_completed_total"] + 1,
        )
        self.assertEqual(snapshot["generation_in_flight"], 0)

    async def test_503_overload_is_tracked_separately(self):
        before = main.service_metrics.snapshot()
        slots = threading.BoundedSemaphore(1)
        self.assertTrue(slots.acquire(blocking=False))
        try:
            with (
                patch.object(main, "inference_slots", slots),
                patch.object(main, "generator", WorkingGenerator()),
            ):
                status, headers, body = await asgi_request(
                    "/generate",
                    "POST",
                    {"prompt": "test", "max_new_tokens": 1},
                )
        finally:
            slots.release()

        self.assertEqual(status, 503)
        self.assertEqual(headers[b"retry-after"], b"1")
        self.assertIn(b"x-request-id", headers)
        self.assertIn("capacity exceeded", body["detail"])
        after = main.service_metrics.snapshot()
        self.assertEqual(
            after["generation_rejected_503_total"],
            before["generation_rejected_503_total"] + 1,
        )
        self.assertEqual(after["generation_in_flight"], 0)

    async def test_inference_failure_returns_safe_500_and_releases_slot(self):
        before = main.service_metrics.snapshot()
        with patch.object(main, "generator", BrokenGenerator()):
            with self.assertLogs("inference_service", level="INFO") as captured:
                status, headers, body = await asgi_request(
                    "/generate",
                    "POST",
                    {"prompt": "secret user prompt", "max_new_tokens": 1},
                )

        self.assertEqual(status, 500)
        self.assertEqual(body, {"detail": "Internal server error"})
        self.assertIn(b"x-request-id", headers)
        self.assertNotIn("secret user prompt", " ".join(captured.output))
        self.assertIn("RuntimeError", " ".join(captured.output))
        after = main.service_metrics.snapshot()
        self.assertEqual(
            after["generation_server_errors_total"],
            before["generation_server_errors_total"] + 1,
        )
        self.assertEqual(after["generation_in_flight"], 0)

        with patch.object(main, "generator", WorkingGenerator()):
            status, _, _ = await asgi_request(
                "/generate",
                "POST",
                {"prompt": "next request", "max_new_tokens": 1},
            )
        self.assertEqual(status, 200)

    async def test_validation_error_counted_as_client_error(self):
        before = main.service_metrics.snapshot()
        status, headers, _ = await asgi_request(
            "/generate",
            "POST",
            {"prompt": "invalid", "max_new_tokens": 999},
        )
        self.assertEqual(status, 422)
        self.assertIn(b"x-request-id", headers)
        after = main.service_metrics.snapshot()
        self.assertEqual(
            after["generation_client_errors_total"],
            before["generation_client_errors_total"] + 1,
        )


class MetricsTests(unittest.TestCase):
    def test_metrics_are_consistent(self):
        metrics = ServiceMetrics()
        metrics.admitted()
        metrics.record_inference(30.0, 20.0)
        metrics.record_http_response(200, 60.0)
        metrics.record_http_response(503, 5.0)
        metrics.record_http_response(422, 2.0)
        metrics.record_http_response(500, 2.0)
        metrics.released()
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["generation_requests_total"], 4)
        self.assertEqual(snapshot["generation_success_total"], 1)
        self.assertEqual(snapshot["generation_rejected_503_total"], 1)
        self.assertEqual(snapshot["generation_client_errors_total"], 1)
        self.assertEqual(snapshot["generation_server_errors_total"], 1)
        self.assertEqual(snapshot["generation_in_flight"], 0)
        self.assertEqual(snapshot["avg_inference_ms"], 30.0)
        self.assertEqual(snapshot["avg_lock_wait_ms"], 20.0)
        self.assertEqual(snapshot["avg_generation_http_latency_ms"], 17.25)


if __name__ == "__main__":
    unittest.main()
