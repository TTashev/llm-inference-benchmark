"""Send HTTP requests at fixed arrival rates to measure inference overload.

Unlike benchmark.py's closed-loop concurrency tests, this client schedules
requests at absolute monotonic times, regardless of previous responses.
"""

import argparse
import csv
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from benchmark import (
    RequestResult,
    nonnegative_int,
    percentile,
    positive_int,
    send_request,
    summarize,
)


@dataclass(frozen=True)
class TimedRequest:
    result: RequestResult
    scheduled_at: float
    started_at: float


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and positive")
    return number


def nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Value must be finite and non-negative")
    return number


def parse_rates(value: str) -> list[float]:
    try:
        rates = [positive_float(part.strip()) for part in value.split(",")]
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated positive rates (e.g. 20,30,40,60)"
        ) from exc
    if not rates or len(rates) != len(set(rates)):
        raise argparse.ArgumentTypeError("Rates must be distinct positive numbers")
    return rates


def request_offsets(rate: float, duration_s: float):
    """Yield scheduled request offsets in [0, duration_s), in seconds."""
    index = 0
    while index / rate < duration_s:
        yield index / rate
        index += 1


def timed_send(
    url: str, payload: dict, scheduled_at: float, worker_slots: threading.Semaphore
) -> TimedRequest:
    started_at = time.perf_counter()
    try:
        result = send_request(url, payload)
        return TimedRequest(result, scheduled_at, started_at)
    finally:
        worker_slots.release()


def summarize_rate(
    completed: list[TimedRequest],
    *,
    rate: float,
    duration_s: float,
    workers: int,
    run: int,
    payload: dict,
    url: str,
    start: float,
    elapsed_s: float,
    scheduled: int,
    client_skipped: int,
) -> dict:
    """Combine server outcomes with client-side rate-generator fidelity."""
    base = summarize(
        [sample.result for sample in completed],
        duration_s=elapsed_s,
        concurrency=workers,
        run=run,
        payload=payload,
        url=url,
    )
    # In rate-controlled tests, workers are client capacity, not HTTP concurrency.
    base.pop("concurrency")

    start_lags = [
        max(0.0, (sample.started_at - sample.scheduled_at) * 1000)
        for sample in completed
    ]
    actual_starts = sum(
        1 for sample in completed if start <= sample.started_at < start + duration_s
    )
    return {
        "mode": "rate",
        "target_rate_rps": rate,
        "target_duration_s": duration_s,
        "client_workers": workers,
        "scheduled_requests": scheduled,
        "client_skipped": client_skipped,
        "observed_start_rps": round(actual_starts / duration_s, 2),
        "p95_start_lag_ms": round(percentile(start_lags, 95), 2) if start_lags else None,
        "max_start_lag_ms": round(max(start_lags), 2) if start_lags else None,
        **base,
    }


def run_rate_benchmark(
    url: str,
    payload: dict,
    rate: float,
    duration_s: float,
    workers: int,
    run: int,
) -> dict:
    """Schedule independent arrivals; never silently queue inside the client."""
    worker_slots = threading.BoundedSemaphore(workers)
    futures = []
    scheduled = 0
    client_skipped = 0
    start = time.perf_counter()
    end_of_window = start + duration_s

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in request_offsets(rate, duration_s):
            scheduled_at = start + offset
            wait_s = scheduled_at - time.perf_counter()
            if wait_s > 0:
                time.sleep(wait_s)

            scheduled += 1
            if not worker_slots.acquire(blocking=False):
                # Client is saturated: count this separately from server HTTP 503.
                client_skipped += 1
                continue

            try:
                futures.append(
                    executor.submit(
                        timed_send, url, payload, scheduled_at, worker_slots
                    )
                )
            except BaseException:
                worker_slots.release()
                raise

        # Include the full requested arrival window before measuring elapsed time.
        remaining = end_of_window - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

        # Drain requests still in flight after scheduling ends.
        completed = [future.result() for future in as_completed(futures)]

    elapsed_s = time.perf_counter() - start
    row = summarize_rate(
        completed,
        rate=rate,
        duration_s=duration_s,
        workers=workers,
        run=run,
        payload=payload,
        url=url,
        start=start,
        elapsed_s=elapsed_s,
        scheduled=scheduled,
        client_skipped=client_skipped,
    )

    print(f"\nRun {run} | target {rate:g} req/s for {duration_s:g}s")
    print(
        f"Scheduled: {scheduled} | sent: {row['requests']} | "
        f"client skipped: {client_skipped} | "
        f"observed send-start rate: {row['observed_start_rps']:.2f} req/s"
    )
    print(
        f"Successful: {row['successful']} | HTTP 503: {row['rejected_503']} "
        f"({row['rejection_rate_pct']:.2f}%) | other errors: {row['errors']}"
    )
    print(
        f"Successful throughput (including drain): {row['throughput_rps']:.2f} req/s"
    )
    if row["successful"]:
        print(
            "Successful latency: "
            f"avg {row['avg_latency_ms']:.2f} ms | "
            f"p50 {row['p50_latency_ms']:.2f} ms | "
            f"p95 {row['p95_latency_ms']:.2f} ms | "
            f"p99 {row['p99_latency_ms']:.2f} ms"
        )
    if row["rejected_503"]:
        print(f"Avg HTTP 503 latency: {row['avg_rejection_latency_ms']:.2f} ms")
    if row["p95_start_lag_ms"] is not None:
        print(
            f"Client dispatch lag: p95 {row['p95_start_lag_ms']:.2f} ms | "
            f"max {row['max_start_lag_ms']:.2f} ms"
        )

    if client_skipped or (
        row["p95_start_lag_ms"] is not None
        and row["p95_start_lag_ms"] > 1000.0 / rate
    ):
        print(
            "WARNING: Client could not maintain the requested rate reliably. "
            "Reduce the target rate or increase --workers."
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument(
        "--rates", type=parse_rates, default=[20.0, 30.0, 40.0, 60.0],
        help="Comma-separated target arrivals per second (default: 20,30,40,60)",
    )
    parser.add_argument(
        "--duration", type=positive_float, default=10.0,
        help="Seconds to schedule requests at each rate (default: 10)",
    )
    parser.add_argument(
        "--workers", type=positive_int, default=64,
        help="Maximum simultaneous client HTTP calls (default: 64)",
    )
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument("--warmup", type=nonnegative_int, default=5)
    parser.add_argument(
        "--pause", type=nonnegative_float, default=1.0,
        help="Idle seconds between rate tests (default: 1)",
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=positive_int, default=20)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results_rate.csv"))
    args = parser.parse_args()

    if args.max_new_tokens > 256:
        parser.error("--max-new-tokens must be <= 256 for this server")

    payload = {"prompt": args.prompt, "max_new_tokens": args.max_new_tokens}
    for index in range(args.warmup):
        result = send_request(args.url, payload)
        if result.error is not None or result.rejected:
            reason = "HTTP 503: overloaded" if result.rejected else result.error
            parser.error(f"Warm-up request {index + 1} failed: {reason}")

    rows = []
    for run in range(1, args.runs + 1):
        for rate in args.rates:
            if rows and args.pause:
                time.sleep(args.pause)
            rows.append(
                run_rate_benchmark(
                    args.url, payload, rate, args.duration, args.workers, run
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rate-test summaries to {args.output}")


if __name__ == "__main__":
    main()
