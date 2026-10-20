"""Benchmark the local inference API under configurable client concurrency."""

import argparse
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests


@dataclass(frozen=True)
class RequestResult:
    latency_s: float
    inference_s: float | None = None
    lock_wait_s: float | None = None
    error: str | None = None


def percentile(values: list[float], percent: float) -> float:
    """Return a percentile using linear interpolation between sorted samples."""
    if not values:
        raise ValueError("Cannot calculate a percentile without samples")
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def parse_concurrency(value: str) -> list[int]:
    try:
        levels = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers (e.g. 1,2,4,8)") from exc

    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("Concurrency levels must be positive integers")
    if len(levels) != len(set(levels)):
        raise argparse.ArgumentTypeError("Concurrency levels must be unique")
    return levels


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative")
    return number


def send_request(url: str, payload: dict) -> RequestResult:
    start = time.perf_counter()

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        inference_ms = float(data["inference_time_ms"])
        lock_wait_ms = float(data["lock_wait_ms"])
        if inference_ms < 0 or lock_wait_ms < 0:
            raise ValueError("Server timing values cannot be negative")

        return RequestResult(
            latency_s=time.perf_counter() - start,
            inference_s=inference_ms / 1000.0,
            lock_wait_s=lock_wait_ms / 1000.0,
        )
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        return RequestResult(
            latency_s=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def summarize(
    results: list[RequestResult],
    duration_s: float,
    concurrency: int,
    run: int,
    payload: dict,
    url: str,
) -> dict:
    successful = [result for result in results if result.error is None]
    latency = [result.latency_s for result in successful]

    def milliseconds(seconds: float | None) -> float | None:
        return round(seconds * 1000, 2) if seconds is not None else None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "concurrency": concurrency,
        "requests": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "duration_s": round(duration_s, 3),
        "throughput_rps": round(len(successful) / duration_s, 2) if duration_s > 0 else 0.0,
        "avg_latency_ms": milliseconds(statistics.mean(latency)) if latency else None,
        "p50_latency_ms": milliseconds(percentile(latency, 50)) if latency else None,
        "p95_latency_ms": milliseconds(percentile(latency, 95)) if latency else None,
        "p99_latency_ms": milliseconds(percentile(latency, 99)) if latency else None,
        "avg_inference_ms": milliseconds(statistics.mean(result.inference_s for result in successful)) if successful else None,
        "avg_lock_wait_ms": milliseconds(statistics.mean(result.lock_wait_s for result in successful)) if successful else None,
        "avg_other_ms": milliseconds(
            statistics.mean(
                result.latency_s - result.inference_s - result.lock_wait_s
                for result in successful
            )
        ) if successful else None,
        "max_latency_ms": milliseconds(max(latency)) if latency else None,
        "max_new_tokens": payload["max_new_tokens"],
        "url": url,
    }


def run_benchmark(
    url: str,
    payload: dict,
    requests_count: int,
    concurrency: int,
    run: int,
) -> dict:
    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, url, payload)
            for _ in range(requests_count)
        ]
        results = [future.result() for future in as_completed(futures)]

    duration_s = time.perf_counter() - start_total
    summary = summarize(results, duration_s, concurrency, run, payload, url)

    print(f"\nRun {run} | concurrency {concurrency} | {summary['successful']}/{requests_count} succeeded")
    print(f"Throughput: {summary['throughput_rps']:.2f} req/s")
    if summary["successful"]:
        print(
            "Client latency: "
            f"avg {summary['avg_latency_ms']:.2f} ms | "
            f"p50 {summary['p50_latency_ms']:.2f} ms | "
            f"p95 {summary['p95_latency_ms']:.2f} ms | "
            f"p99 {summary['p99_latency_ms']:.2f} ms"
        )
        print(
            f"Avg inference: {summary['avg_inference_ms']:.2f} ms | "
            f"lock wait: {summary['avg_lock_wait_ms']:.2f} ms | "
            f"other: {summary['avg_other_ms']:.2f} ms"
        )
    if summary["errors"]:
        example = next(result.error for result in results if result.error is not None)
        print(f"Errors: {summary['errors']} (example: {example})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--requests", type=positive_int, default=50, help="Requests per concurrency level")
    parser.add_argument("--concurrency", type=parse_concurrency, default=[1, 2, 4, 8],
                        help="Comma-separated client counts (default: 1,2,4,8)")
    parser.add_argument("--runs", type=positive_int, default=1, help="Repeated runs per level")
    parser.add_argument("--warmup", type=nonnegative_int, default=5)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=positive_int, default=20)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"))
    args = parser.parse_args()

    payload = {"prompt": args.prompt, "max_new_tokens": args.max_new_tokens}
    for index in range(args.warmup):
        result = send_request(args.url, payload)
        if result.error is not None:
            parser.error(f"Warm-up request {index + 1} failed: {result.error}")

    rows = []
    for run in range(1, args.runs + 1):
        for concurrency in args.concurrency:
            rows.append(run_benchmark(args.url, payload, args.requests, concurrency, run))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} benchmark summaries to {args.output}")
    print("Percentiles use linear interpolation; p99 needs many requests to be meaningful.")


if __name__ == "__main__":
    main()
