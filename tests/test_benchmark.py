import argparse
import unittest
from unittest.mock import Mock, patch

from benchmark import RequestResult, parse_concurrency, percentile, send_request, summarize


class BenchmarkMetricTests(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self):
        samples = [50.0, 10.0, 40.0, 20.0, 30.0]
        self.assertEqual(percentile(samples, 50), 30.0)
        self.assertAlmostEqual(percentile(samples, 95), 48.0)
        self.assertAlmostEqual(percentile(samples, 99), 49.6)
        self.assertEqual(percentile([7.0], 99), 7.0)

    def test_percentile_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            percentile([], 95)

    def test_concurrency_validation(self):
        self.assertEqual(parse_concurrency("1,2,4,8"), [1, 2, 4, 8])
        for invalid in ("", "0,2", "1,1", "two,4"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_concurrency(invalid)

    def test_summary_excludes_failures_from_latency_metrics(self):
        results = [
            RequestResult(latency_s=0.1, inference_s=0.04, lock_wait_s=0.03),
            RequestResult(latency_s=0.2, inference_s=0.05, lock_wait_s=0.1),
            RequestResult(latency_s=0.12, error="HTTPError"),
        ]
        summary = summarize(
            results,
            duration_s=0.5,
            concurrency=2,
            run=1,
            payload={"max_new_tokens": 20},
            url="http://127.0.0.1:8000/generate",
        )
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["throughput_rps"], 4.0)
        self.assertEqual(summary["p50_latency_ms"], 150.0)
        self.assertEqual(summary["p95_latency_ms"], 195.0)
        self.assertEqual(summary["avg_inference_ms"], 45.0)
        self.assertEqual(summary["avg_lock_wait_ms"], 65.0)
        self.assertEqual(summary["avg_other_ms"], 40.0)


    def test_rejected_requests_excluded_from_success_latency(self):
        results = [
            RequestResult(latency_s=0.1, inference_s=0.04, lock_wait_s=0.02),
            RequestResult(latency_s=0.005, rejected=True),
            RequestResult(latency_s=0.006, rejected=True),
            RequestResult(latency_s=0.2, error="ConnectionError"),
        ]
        summary = summarize(
            results,
            duration_s=0.5,
            concurrency=4,
            run=1,
            payload={"max_new_tokens": 20},
            url="http://127.0.0.1:8000/generate",
        )
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["rejected_503"], 2)
        self.assertEqual(summary["rejection_rate_pct"], 50.0)
        self.assertEqual(summary["avg_rejection_latency_ms"], 5.5)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["throughput_rps"], 2.0)
        self.assertEqual(summary["p50_latency_ms"], 100.0)
        self.assertEqual(summary["p99_latency_ms"], 100.0)

    def test_send_request_classifies_503_as_rejection(self):
        response = Mock()
        response.status_code = 503
        with patch("benchmark.requests.post", return_value=response):
            result = send_request(
                "http://127.0.0.1:8000/generate",
                {"prompt": "test", "max_new_tokens": 1},
            )
        self.assertTrue(result.rejected)
        self.assertIsNone(result.error)
        self.assertIsNone(result.inference_s)
        response.raise_for_status.assert_not_called()



if __name__ == "__main__":
    unittest.main()
