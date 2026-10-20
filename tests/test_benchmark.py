import argparse
import unittest

from benchmark import RequestResult, parse_concurrency, percentile, summarize


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


if __name__ == "__main__":
    unittest.main()
