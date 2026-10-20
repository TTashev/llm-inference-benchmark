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

The model is loaded **once at application startup** and reused for incoming requests. The `/generate` handler is synchronous; FastAPI executes synchronous handlers in a thread pool. An explicit lock serializes access to the shared text-generation pipeline for our initial, controlled concurrency experiments.

## Features and branch status

The following are available on `main`:

- `GET /health` — basic health endpoint.
- `POST /generate` — generate text using `sshleifer/tiny-gpt2`.
- Pydantic validation for `prompt` and `max_new_tokens` (1–256, default 50).
- Model loaded once at startup, CPU execution with one PyTorch thread, and serialized inference using a shared lock.
- Response timings: `inference_time_ms` (model pipeline execution inside the lock) and `lock_wait_ms` (time waiting to acquire the lock).
- Baseline benchmark: 50 requests at each concurrency level (1, 2, 4, 8), showing throughput, successful/failed requests, mean client latency, inference time, lock wait, and remaining overhead.

**Important branch note (October 2026):** Both [PR #1](https://github.com/TTashev/llm-inference-benchmark/pull/1) and [PR #2](https://github.com/TTashev/llm-inference-benchmark/pull/2) are marked merged, but PR #2 was merged into `feature/lock-wait-metrics` *after* that branch had already been merged into `main`. Consequently, `main` **does not yet include** PR #2's benchmark improvements. The enhanced benchmark and its tests currently live on `feature/lock-wait-metrics`:

- p50, p95, and p99 client-latency percentiles (linear interpolation)
- Configurable number of requests, concurrency levels, repeats, warm-up, prompt, model token count, URL, and output file
- CSV summaries (one row per run and concurrency setting)
- Unit tests for percentile calculations, argument validation, and successful/failed request accounting

See [Enhanced benchmark](#enhanced-benchmark-feature-branch) for usage. A follow-up integration into `main` is still required.

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

Generate text (Windows PowerShell):

```powershell
$payload = @{ prompt = "Once upon a time"; max_new_tokens = 20 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -ContentType "application/json" -Body $payload
```

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

## Running the benchmark

Leave the inference server running and open a second terminal in the repository.

### Baseline benchmark (`main`)

Linux (with virtual environment activated):

```bash
python benchmark.py
```

Windows (without activating the virtual environment):

```powershell
.\.venv\Scripts\python.exe benchmark.py
```

The baseline benchmark warms up the model with five requests and sends 50 requests at each concurrency level: 1, 2, 4, and 8. It prints aggregate throughput, mean client latency, mean inference time, mean lock wait, remaining overhead, and failures.

### Enhanced benchmark (feature branch)

To use the new p50/p95/p99 and CSV functionality from merged PR #2, switch to the branch that currently contains it:

```bash
git fetch origin
git switch feature/lock-wait-metrics
```

If that branch is not available locally, use `git switch --track origin/feature/lock-wait-metrics`.

With the server running, execute:

```bash
python benchmark.py --requests 500 --concurrency 1,2,4,8
```

For three repeated runs and a custom CSV destination:

```bash
python benchmark.py --requests 200 --concurrency 1,2,4,8 --runs 3 --output benchmark_results.csv
```

Use `python benchmark.py --help` to see all parameters. Available options include:

| Option | Default | Description |
| --- | --- | --- |
| `--requests` | `50` | Number of requests at each concurrency level per run |
| `--concurrency` | `1,2,4,8` | Comma-separated number of simultaneous benchmark clients |
| `--runs` | `1` | Number of complete benchmark repetitions |
| `--warmup` | `5` | Warm-up requests before measurements |
| `--max-new-tokens` | `20` | Tokens requested per inference call |
| `--prompt` | `Once upon a time` | Generation prompt |
| `--url` | `http://127.0.0.1:8000/generate` | Target endpoint |
| `--output` | `benchmark_results.csv` | CSV summary output path (overwritten on each run) |

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

The enhanced benchmark writes **one CSV summary row per concurrency level and run**, including a timestamp, workload configuration, successes/errors, throughput, mean latency, p50/p95/p99, model inference time, lock wait, and other overhead. With `--runs 3` and four concurrency levels, this produces 12 summary rows.

### Understanding the metrics

| Metric | Meaning |
| --- | --- |
| Throughput (req/s) | Successful responses divided by total elapsed time per workload |
| Mean client latency | Mean end-to-end HTTP request duration for successful requests |
| p50 / p95 / p99 latency | Median, 95th, and 99th percentile client latency (enhanced benchmark only) |
| Inference time | Time spent executing the model pipeline after the inference lock is acquired |
| Lock wait | Time waiting to acquire the shared inference lock |
| Other overhead | Mean client latency minus mean inference time minus mean lock wait (HTTP, scheduling, and remaining overhead) |
| Errors | Requests that fail; excluded from successful-request latency percentiles |

**Note:** More concurrent requests do not necessarily increase throughput when inference is serialized. Percentiles measured over small samples, especially p99, should be interpreted cautiously.

## Model and limitations

- Model: [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2), used to test the serving pipeline rather than to provide useful text generation.
- `torch.set_num_threads(1)` is configured in `main.py` for the current tiny CPU model. This is an experimental configuration, **not a universal recommendation**.
- Inference is serialized by a mutex; this simplifies behavior but limits concurrency.
- Benchmarks are local and short. Results depend on CPU resources, OS, VM configuration, background load, and software versions.
- The service does not yet implement request admission limits, production error handling, authentication, or observability.
- `/health` is a basic liveness check, not a comprehensive readiness check.

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
3. **Benchmarking and metrics** — enhanced features implemented and tested on Windows; **still need integrating into `main`** because PR #2 merged into a feature branch.
4. **Performance investigations** — current experiments cover CPU threading and mutex contention; lock-wait instrumentation is on `main`.
5. **Production-minded improvements** — planned: bounded request admission, overload/backpressure handling, better error reporting, and more testing.

This project is meant to evolve incrementally, with measurements informing each subsequent design decision.
