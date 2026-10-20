import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


URL = "http://127.0.0.1:8000/generate"
PAYLOAD = {
    "prompt": "Once upon a time",
    "max_new_tokens": 20,
}
TOTAL_REQUESTS = 50


def send_request():
    start = time.perf_counter()

    try:
        response = requests.post(URL, json=PAYLOAD, timeout=60)
        response.raise_for_status()

        latency = time.perf_counter() - start
        data = response.json()

        inference_time = data["inference_time_ms"] / 1000.0
        lock_wait = data["lock_wait_ms"] / 1000.0
        return latency, inference_time, lock_wait, True

    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Request failed: {exc}")
        return time.perf_counter() - start, 0.0, 0.0, False


def run_benchmark(concurrency):
    latencies = []
    inference_times = []
    lock_wait_times = []
    errors = 0

    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request)
            for _ in range(TOTAL_REQUESTS)
        ]

        for future in as_completed(futures):
            latency, inference_time, lock_wait, success = future.result()
            if success:
                latencies.append(latency)
                inference_times.append(inference_time)
                lock_wait_times.append(lock_wait)
            else:
                errors += 1

    total_time = time.perf_counter() - start_total

    print(f"\nConcurrency: {concurrency}")
    print(f"Successful requests: {len(latencies)}")
    print(f"Failed requests: {errors}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Throughput: {len(latencies) / total_time:.2f} req/s")

    if latencies:
        avg_latency = statistics.mean(latencies)
        avg_inference = statistics.mean(inference_times)
        avg_lock_wait = statistics.mean(lock_wait_times)
        other_overhead = avg_latency - avg_inference - avg_lock_wait

        print(f"Average client latency: {avg_latency * 1000:.2f} ms")
        print(f"Average inference time: {avg_inference * 1000:.2f} ms")
        print(f"Average lock wait: {avg_lock_wait * 1000:.2f} ms")
        print(f"Other overhead: {other_overhead * 1000:.2f} ms")


if __name__ == "__main__":
    for _ in range(5):
        _, _, _, success = send_request()
        if not success:
            raise RuntimeError("Warm-up failed; check the inference server")

    for concurrency in [1, 2, 4, 8]:
        run_benchmark(concurrency)
