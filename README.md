# LLM Inference Benchmarking Service

A small learning project for exploring **AI model inference**, **HTTP APIs**, **concurrency**, and **performance benchmarking** using Python, FastAPI, PyTorch, and Hugging Face Transformers.

The project exposes a local text-generation API backed by a tiny pretrained model and includes a client that measures how response time and throughput change as the number of concurrent clients increases.

> **Learning prototype, not a production inference service.** The model is intentionally tiny and its output is often repetitive or nonsensical. The emphasis is on the serving and performance aspects, not response quality.

## Architecture

```text
Benchmark client (benchmark.py)
        |
        | concurrent HTTP POST /generate
        v
FastAPI application (main.py)
        |
        | request validation (Pydantic)
        v
Admission limit (MAX_IN_FLIGHT, default 4)
        |                    \\
        | admitted            \\ full -> HTTP 503 + Retry-After
        v
Worker thread pool
        |
        v
Shared inference lock
        |
        | one inference call at a time
        v
Hugging Face text-generation pipeline
        |
        | Tiny GPT-2 + PyTorch on CPU
        v
JSON response (generated text + timing)
```

The model is loaded **once at application startup** and reused for incoming requests. The `/generate` handler is asynchronous so it can make a **non-blocking admission decision before submitting the blocking model call to the worker thread pool**. An explicit lock still serializes access to the shared text-generation pipeline.

For each server process, `MAX_IN_FLIGHT` (default **4**) limits the number of admitted generation requests, including both **1 running inference** and requests waiting for the inference lock. When all permits are occupied, the server immediately responds with **HTTP 503 Service Unavailable**, `Retry-After: 1`, and a JSON error instead of accepting another inference request.

## Features

The following functionality is available on `main`:

- `GET /health` — basic health endpoint.
- `POST /generate` — generate text using `sshleifer/tiny-gpt2`.
- Pydantic validation for `prompt` and `max_new_tokens` (1–256, default 50).
- Model loaded once at startup, CPU inference with one PyTorch thread, and shared-lock serialization.
- Response timings: `inference_time_ms` (inside the lock) and `lock_wait_ms` (waiting to acquire the lock).
- Benchmark metrics: request counts, failures, throughput, average latency, p50/p95/p99 client latency, average inference time, lock wait, and residual overhead.
- Configurable workloads: request count, concurrency levels, repeated runs, warm-up, prompt, token count, endpoint, and CSV destination.
- CSV summary export and unit tests for benchmark calculations.
- **Milestone 4 (on `main`):** bounded inference admission, HTTP 503 overload rejection, and rate-controlled benchmarking.
- **Reliability & observability (feature branch):** JSON request logs, request IDs, liveness/readiness endpoints, in-process metrics, and sanitized HTTP 500 responses.

The benchmark improvements were developed in [PR #2](https://github.com/TTashev/llm-inference-benchmark/pull/2); lock-wait instrumentation was developed in [PR #1](https://github.com/TTashev/llm-inference-benchmark/pull/1).

## Requirements

- Python **3.11+** recommended
- Windows, Linux, or a Linux VM
- Internet connection for the initial Hugging Face model download
- CPU inference; **no GPU is required**

## Setup

Clone the repository:

```bash
git clone https://github.com/TTashev/llm-inference-benchmark.git
cd llm-inference-benchmark
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install fastapi uvicorn transformers requests
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn transformers requests
```

Using `.venv\Scripts\python.exe` directly avoids needing to change PowerShell script execution policies. If Python 3.11 is not available, create the environment with an installed compatible Python version instead.

## Running the inference server

From the project directory, with the environment active on Linux:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

On Windows, if you did not activate the environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The first startup may download the model files. Run a **single Uvicorn worker** for the baseline experiments; multiple workers would each load their own model instance.

Once ready, browse to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API documentation.

### Example requests

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Generate text (Linux / Bash):

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_new_tokens":20}'
```

Generate text (Windows PowerShell, using `Invoke-RestMethod`):

```powershell
$payload = @{
    prompt = "Once upon a time"
    max_new_tokens = 20
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/generate" `
    -Method Post `
    -ContentType "application/json" `
    -Body $payload
```

Run this in a second PowerShell terminal while Uvicorn is listening on `127.0.0.1:8000`. The trailing backtick on each continued line must be the last character (no spaces after it).

A successful JSON response includes:

```json
{
  "generated_text": "Once upon a time ...",
  "max_new_tokens": 20,
  "inference_time_ms": 30.0,
  "lock_wait_ms": 0.0
}
```

The text and timing shown above are **illustrative**, not guaranteed output.

## Reliability and observability (new feature branch)

On `feature/reliability-observability`, the inference server gains:

- **`GET /health`** — lightweight liveness probe, returns `{"status":"ok"}` even when the model isn't loaded.
- **`GET /ready`** — readiness probe, returns HTTP 200 after the model pipeline loads; HTTP 503 if the pipeline is unavailable.
- **`GET /metrics`** — thread-safe, **per-process** JSON counters for generation outcomes (success, intentional 503, client 4xx, unexpected server 5xx), current admitted requests, uptime, and average inference/lock-wait/HTTP times.
- **`X-Request-ID`** — server-generated correlation ID returned in each HTTP response.
- **Structured JSON log events** — request method, known route, response status, request ID, elapsed milliseconds, and model lifecycle events. Prompt bodies and generated text are **not logged**; unrecognized URL paths are redacted.
- **Sanitized HTTP 500 responses** — unexpected errors before a response starts return `{"detail":"Internal server error"}`, with an `X-Request-ID` header. Internal exception text is not exposed to clients; JSON exception logs include only the exception type.

From PowerShell, start a **single-worker** server:

```powershell
$env:MAX_IN_FLIGHT = "4"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then in another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/metrics
(Invoke-WebRequest http://127.0.0.1:8000/health).Headers["X-Request-ID"]
```

A typical **request log event** (example values are illustrative):

```json
{"timestamp_utc":"2026-10-20T10:00:00+00:00","event":"http_request","request_id":"0123456789abcdef0123456789abcdef","method":"POST","path":"/generate","status_code":200,"duration_ms":42.5}
```

To avoid per-request logging overhead during throughput benchmarks, set `$env:LOG_REQUESTS = "0"` in the **server** terminal before starting Uvicorn. Metrics, request IDs, startup logs, and exception events remain enabled. Remove the variable (or set it to `1`) to restore access logs.

The `/metrics` fields are cumulative since process startup; they reset when Uvicorn restarts. HTTP outcome counters cover **`/generate` only**, and `avg_generation_http_latency_ms` averages all its HTTP outcomes (including rejected/errors); it is **not** the same as successful-only latency in benchmark CSVs. `generation_in_flight` includes admitted calls waiting for the inference mutex, not all open TCP connections. `inference_completed_total` and its average timings count successful model executions.

**Security/operations limitations:** `/metrics` is unauthenticated, so keep this learning service bound to `127.0.0.1` and protect monitoring endpoints before any external deployment. These are in-memory counters, not persistent time series or Prometheus histograms. Monitoring and admission are per Uvicorn worker, not cluster-wide. A failure after response headers were sent cannot safely be converted to a new 500 response.

Run the complete test suite after checking out this feature branch:

```powershell
python -m unittest discover -s tests -v
```

## Running the benchmark

Leave the inference server running and open a second terminal in the repository.

### Benchmark usage

Linux (with virtual environment activated):

```bash
python benchmark.py
```

Windows (without activating the virtual environment):

```powershell
.\.venv\Scripts\python.exe benchmark.py
```

By default, the client warms up with five requests, then sends 50 requests at each concurrency level (1, 2, 4, and 8).

For a larger workload:

```bash
python benchmark.py --requests 500 --concurrency 1,2,4,8
```

For three repetitions per concurrency level:

```bash
python benchmark.py --requests 200 --concurrency 1,2,4,8 --runs 3 --output benchmark_results.csv
```

Use `python benchmark.py --help` to see the full CLI. Options include:

| Option | Default | Description |
| --- | --- | --- |
| `--requests` | `50` | Requests per concurrency level and run |
| `--concurrency` | `1,2,4,8` | Comma-separated simultaneous client counts |
| `--runs` | `1` | Number of repeated benchmark runs |
| `--warmup` | `5` | Warm-up requests before measurement |
| `--max-new-tokens` | `20` | Generated tokens requested per call |
| `--prompt` | `Once upon a time` | Input prompt |
| `--url` | `http://127.0.0.1:8000/generate` | Target endpoint |
| `--output` | `benchmark_results.csv` | CSV summary file, overwritten each invocation |

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

The CSV contains one summary row per concurrency level and run, with a timestamp, request counts, failures, throughput, average latency, p50/p95/p99, inference time, lock wait, and residual overhead. Four concurrency levels and three runs produce 12 summary rows.

### Overload and backpressure experiment

You can control per-process admission capacity through `MAX_IN_FLIGHT` (**default: 4**). It limits all accepted `/generate` requests, **not** the number of parallel model calls. The model lock still permits only one inference call at a time.

Windows PowerShell — set the limit **in the terminal that starts the server**:

```powershell
$env:MAX_IN_FLIGHT = "4"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Linux:

```bash
MAX_IN_FLIGHT=4 python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

From a second terminal, run the benchmark:

```bash
python benchmark.py --requests 200 --concurrency 1,2,4,8,16,32 --output benchmark_results_overload.csv
```

As load increases beyond admitted capacity, some requests will return **503**, which is an **expected overload response** rather than an unexpected inference failure. The enhanced benchmark reports the number of rejected requests, rejection rate, average rejection latency, successful throughput, and latency percentiles **for successful requests only**. The CSV records `rejected_503`, `rejection_rate_pct`, and `avg_rejection_latency_ms`; the `errors` field now counts other failures separately.

Note that **`benchmark.py`** is a **closed-loop client workload**: fast 503 responses let a worker send another request quickly. Thus, its rejection percentages depend on the load generator and should not be interpreted as an arrival-rate or production capacity estimate. Use the rate-controlled experiment below to study sustained offered load.

This limit applies **per Uvicorn worker process**. To keep the experiment interpretable, run **one Uvicorn worker**. It is not a global limit across multiple workers or servers, and does not bound TCP or web-server queues before FastAPI receives a request. There is no retry mechanism in the benchmark; clients are responsible for respecting `Retry-After` if they retry in a real application.

### Rate-controlled benchmark (fixed arrival rate)

Use `rate_benchmark.py` to test sustained offered load without waiting for a prior HTTP response before scheduling another request. This is more appropriate than the closed-loop `benchmark.py` when investigating backpressure and admission limits.

With a **single-worker** Uvicorn server running and `MAX_IN_FLIGHT=4`, execute in a separate terminal:

```bash
python rate_benchmark.py --rates 20,30,40,60 --duration 10 --output benchmark_results_rate.csv
```

For more stable observations, use 20 seconds per rate, repeat the set three times, and leave the client default of 64 worker threads unless scheduling warnings appear:

```bash
python rate_benchmark.py --rates 20,30,40,60 --duration 20 --runs 3 --workers 64
```

The default settings are `--rates 20,30,40,60`, `--duration 10` seconds, `--workers 64`, `--runs 1`, `--warmup 5`, and `--pause 1` second between rate tests. As with `benchmark.py`, the prompt, target URL, token count, and CSV destination are configurable; see `python rate_benchmark.py --help`.

This client schedules arrivals at absolute monotonic-clock times, with one new request every `1 / rate` seconds. It does **not** automatically retry HTTP 503 rejections. Each rate has a fixed arrival window; requests still running when the window ends are allowed to finish before statistics are printed.

The output and CSV include:
- **`target_rate_rps`** — requested send schedule in requests per second
- **`observed_start_rps`** — number of HTTP calls actually started during the scheduling window divided by its duration
- **`scheduled_requests` / `client_skipped`** — intended arrivals and requests the client could not launch because all of its worker slots were occupied; these are **not** server rejections
- **`p95_start_lag_ms` / `max_start_lag_ms`** — delay between the scheduled request time and when a client thread actually started it
- **`successful`, `rejected_503`, `rejection_rate_pct`, `errors`** — server outcomes among HTTP requests actually sent
- **`throughput_rps`** — successful responses divided by the full benchmark wall time, including time spent draining outstanding requests; compare this alongside offered rate and actual starts
- **`p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`** — latency percentiles of successful requests only

A warning means client dispatch was delayed by more than one arrival interval at p95 or requests were skipped locally. Increase `--workers` or lower `--rates` before interpreting these results as server capacity.

This is a basic **constant-rate open-loop load generator** for local learning, not a production load-testing platform. The Python client, Windows timer granularity, CPU contention (particularly when server and benchmark share the same machine), and short test duration can introduce scheduling jitter. Repeated runs and monitoring system utilization make conclusions more reliable.

### Understanding the metrics

| Metric | Meaning |
| --- | --- |
| Throughput (req/s) | Successful responses divided by total elapsed time per workload |
| Mean client latency | Mean end-to-end HTTP request duration for successful requests |
| p50 / p95 / p99 latency | Median, 95th, and 99th percentile client latency |
| Inference time | Time spent executing the model pipeline after the inference lock is acquired |
| Lock wait | Time waiting to acquire the shared inference lock |
| Other overhead | Mean client latency minus mean inference time minus mean lock wait (HTTP, scheduling, and remaining overhead) |
| Rejected (HTTP 503) | Requests deliberately rejected because inference admission capacity is full (Milestone 4) |
| Rejection rate | Share of all attempted requests receiving HTTP 503 (Milestone 4 feature branch) |
| Avg rejection latency | Mean end-to-end client time for rejected requests (Milestone 4 feature branch) |
| Errors | Other failed requests, separate from HTTP 503 overload responses in Milestone 4; failures are excluded from successful-request latency percentiles |

**Note:** More concurrent requests do not necessarily increase throughput when inference is serialized. Percentiles measured over small samples, especially p99, should be interpreted cautiously.

## Model and limitations

- Model: [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2), used to test the serving pipeline rather than to provide useful text generation.
- `torch.set_num_threads(1)` is configured in `main.py` for the current tiny CPU model. This is an experimental configuration, **not a universal recommendation**.
- Inference is serialized by a mutex; this simplifies behavior but limits concurrency.
- Benchmarks are local and short. Results depend on CPU resources, OS, VM configuration, background load, and software versions.
- The `main` branch includes bounded admission; metrics and structured logs are under review in the reliability/observability feature branch.
- The service is not yet production-hardened: it lacks comprehensive authentication, error handling, monitoring, and distributed admission control.
- On the reliability/observability branch, `/health` is liveness and `/ready` checks whether the model was loaded; this does not guarantee end-to-end inference success.

## Example benchmark observations

The following are **user-reported measurements** from a Windows host (not performance guarantees), using the enhanced benchmark with 500 requests per concurrency setting and one run per setting:

| Concurrent clients | Throughput (req/s) | Mean latency (ms) | p95 latency (ms) | Mean lock wait (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 26.82 | 37.11 | 48.69 | 0.00 |
| 2 | 33.21 | 60.00 | 62.00 | 20.54 |
| 4 | 33.04 | 120.54 | 123.28 | 81.65 |
| 8 | 33.30 | 238.40 | 243.84 | 199.01 |

All 2,000 requests succeeded. Throughput leveled off around 33 req/s while lock-wait and end-to-end latency increased with concurrency. Hardware, software versions, and background activity affect results; repeat tests before drawing robust performance conclusions.

## Project roadmap

1. **HTTP API and validation** — complete on `main`.
2. **CPU model inference** — complete on `main`.
3. **Benchmarking and metrics** — complete: CLI configuration, p50/p95/p99, CSV export, and unit tests.
4. **Performance investigations** — CPU threading and mutex contention measured; lock-wait instrumentation is on `main`.
5. **Overload protection / backpressure** — complete on `main`; PR #4 merged, 14 unit tests passed and rate-controlled overload verified.
6. **Reliability & observability** — JSON access logs, request IDs, readiness, in-process counters and sanitized failure responses implemented in `feature/reliability-observability` for validation; not yet merged.
7. **Further production-minded improvements** — future: richer monitoring, authentication, dashboards, and multi-worker behavior.

This project is meant to evolve incrementally, with measurements informing each subsequent design decision.
