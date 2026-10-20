"""Unit tests for rate-controlled load generation and result reporting."""

import argparse
import threading
import unittest
from unittest.mock import patch

from benchmark import RequestResult
from rate_benchmark import (
    TimedRequest,
    parse_rates,
    positive_float,
    request_offsets,
    summarize_rate,
    timed_send,
)


class RateBenchmarkTests(unittest.TestCase):
    def test_rates_validate_positive_finite_distinct_numbers(self):
        self.assertEqual(parse_rates("20, 30.5,60"), [20.0, 30.5, 60.0])
        for invalid in ("", "0", "-1", "1,1.0", "NaN", "inf", "twenty"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_rates(invalid)

        for invalid in ("0", "-5", "nan", "inf"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_float(invalid)

    def test_rate_schedule_uses_fixed_deadlines(self):
        self.assertEqual(list(request_offsets(2.0, 1.0)), [0.0, 0.5])
        self.assertEqual(list(request_offsets(4.0, 0.5)), [0.0, 0.25])
        self.assertEqual(list(request_offsets(1.0, 0.1)), [0.0])

    def test_summary_separates_503_from_errors_and_client_skips(self):
        start = 100.0
        completed = [
            TimedRequest(
                RequestResult(latency_s=0.1, inference_s=0.05, lock_wait_s=0.02),
                scheduled_at=100.0,
                started_at=100.002,
            ),
            TimedRequest(
                RequestResult(latency_s=0.005, rejected=True),
                scheduled_at=100.2,
                started_at=100.203,
            ),
            TimedRequest(
                RequestResult(latency_s=0.01, error="connection refused"),
                scheduled_at=100.4,
                started_at=100.404,
            ),
        ]
        row = summarize_rate(
            completed,
            rate=5.0,
            duration_s=1.0,
            workers=4,
            run=1,
            payload={"max_new_tokens": 20},
            url="http://127.0.0.1:8000/generate",
            start=start,
            elapsed_s=1.1,
            scheduled=4,
            client_skipped=1,
        )
        self.assertEqual(row["target_rate_rps"], 5.0)
        self.assertEqual(row["scheduled_requests"], 4)
        self.assertEqual(row["requests"], 3)
        self.assertEqual(row["client_skipped"], 1)
        self.assertEqual(row["successful"], 1)
        self.assertEqual(row["rejected_503"], 1)
        self.assertEqual(row["errors"], 1)
        self.assertEqual(row["rejection_rate_pct"], 33.33)
        self.assertEqual(row["observed_start_rps"], 3.0)
        self.assertEqual(row["p95_start_lag_ms"], 3.9)
        self.assertEqual(row["max_start_lag_ms"], 4.0)
        self.assertEqual(row["p95_latency_ms"], 100.0)
        self.assertNotIn("concurrency", row)

    def test_worker_slot_released_even_when_request_raises(self):
        slots = threading.BoundedSemaphore(1)
        self.assertTrue(slots.acquire(blocking=False))
        with patch("rate_benchmark.send_request", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                timed_send("http://localhost", {}, 1.0, slots)
        self.assertTrue(slots.acquire(blocking=False))
        slots.release()


if __name__ == "__main__":
    unittest.main()
