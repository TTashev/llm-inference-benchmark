
import time
import statistics
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8000/generate"

PAYLOAD = {
    "prompt": "Once upon a time",
    "max_new_tokens": 20
}

TOTAL_REQUESTS = 50


def send_request():
    start = time.perf_counter()

    try:
        response = requests.post(
            URL,
            json=PAYLOAD,
            timeout=60
        )
        response.raise_for_status()

        latency = time.perf_counter() - start

        data = response.json()
        inference_time = data["inference_time_ms"] / 1000.0

        return latency, inference_time, True

    except (requests.RequestException, ValueError, KeyError):
        return time.perf_counter() - start, 0.0, False


def run_benchmark(concurrency):
    latencies = []
    inference_times = []
    errors = 0

    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request)
            for _ in range(TOTAL_REQUESTS)
        ]

        for future in as_completed(futures):
            latency, inference_time, success = future.result()

            if success:
                latencies.append(latency)
                inference_times.append(inference_time)
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

        print(f"Average client latency: {avg_latency * 1000:.2f} ms")
        print(f"Average inference time: {avg_inference * 1000:.2f} ms")
        print(
            f"Average other overhead: "
            f"{(avg_latency - avg_inference) * 1000:.2f} ms"
        )


if __name__ == "__main__":
    # Warm up the model first
    for _ in range(5):
        send_request()

    for concurrency in [1, 2, 4, 8]:
        run_benchmark(concurrency)
