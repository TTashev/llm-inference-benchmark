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

## Features on `main`

- `GET /health` — basic health endpoint.
- `POST /generate` — accepts a prompt and generates text using `sshleifer/tiny-gpt2`.
- Pydantic request validation, including `max_new_tokens` between **1 and 256** (default **50**).
- Server-side `inference_time_ms` measurement covering the pipeline call **after acquiring the lock**.
- A benchmark client that sends **50 requests** at each of **1, 2, 4, and 8** concurrent clients.
- Benchmark reporting for successful/failed requests, throughput, average client latency, average model inference time, and other combined overhead.

**Work in progress (not yet on `main`):**
- [PR #1 — mutex lock-wait metrics](https://github.com/TTashev/llm-inference-benchmark/pull/1) adds `lock_wait_ms`.
- [PR #2 — configurable benchmark metrics and CSV export](https://github.com/TTashev/llm-inference-benchmark/pull/2) adds p50/p95/p99, configurable runs, and CSV summaries. It currently builds on PR #1.

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
  "inference_time_ms": 30.0
}
```

The text and timing shown above are **illustrative**, not guaranteed output.

## Running the benchmark

Leave the inference server running and open a second terminal in the project directory.

Linux (with virtual environment active):

```bash
python benchmark.py
```

Windows (without virtual environment activation):

```powershell
.\.venv\Scripts\python.exe benchmark.py
```

On `main`, the client warms up the server with five requests and then sends 50 requests for each concurrency level (1, 2, 4, 8).

The metrics are:

| Metric | Meaning |
| --- | --- |
| Throughput (req/s) | Successful responses divided by elapsed benchmark duration |
| Client latency | End-to-end time for an individual HTTP request |
| Inference time | Time spent executing the model pipeline inside the inference lock |
| Other overhead | Client latency minus reported inference time; includes waiting for the lock, networking and scheduling |
| Failed requests | Requests with an HTTP/connection error or invalid response |

A key learning goal is to distinguish **more requests in flight** from **more actual inference throughput**. A shared inference lock may cause throughput to plateau while latency rises with client concurrency.

## Model and limitations

- Model: [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2), used to test the serving pipeline rather than to provide useful text generation.
- `torch.set_num_threads(1)` is configured in `main.py` for the current tiny CPU model. This is an experimental configuration, **not a universal recommendation**.
- Inference is serialized by a mutex; this simplifies behavior but limits concurrency.
- Benchmarks are local and short. Results depend on CPU resources, OS, VM configuration, background load, and software versions.
- The service does not yet implement request admission limits, production error handling, authentication, or observability.
- `/health` is a basic liveness check, not a comprehensive readiness check.

## Project roadmap

1. **HTTP API and validation** — completed.
2. **CPU model inference** — completed.
3. **Benchmarking and metrics** — baseline completed; percentile metrics, configurable workloads, and CSV export are in [PR #2](https://github.com/TTashev/llm-inference-benchmark/pull/2).
4. **Performance investigations** — measuring inference threading and lock contention; detailed mutex-wait metrics are in [PR #1](https://github.com/TTashev/llm-inference-benchmark/pull/1).
5. **Production-minded improvements** — planned: overload protection, bounded request admission, improved error handling, and further testing.

This project is meant to evolve incrementally, with measurements informing each subsequent design decision.
