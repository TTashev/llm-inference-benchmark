"""Tiny CPU inference service with admission control, dynamic batching and
observability."""

import asyncio
import os
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from batched_inference import GreedyBatchGenerator, MicroBatcher
from observability import (
    RequestObservabilityMiddleware,
    ServiceMetrics,
    log_event,
    request_logging_enabled,
)
from pydantic import BaseModel, Field
from transformers import pipeline


def read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def read_max_in_flight() -> int:
    """Maximum admitted requests, including running and queued inference."""
    return read_positive_int_env("MAX_IN_FLIGHT", 64)


def read_max_batch_size() -> int:
    """Maximum number of queued requests decoded together in one batch."""
    return read_positive_int_env("MAX_BATCH_SIZE", 32)


MAX_IN_FLIGHT = read_max_in_flight()
MAX_BATCH_SIZE = read_max_batch_size()
generator = None
# Non-blocking admission at the async HTTP handler, before work reaches the
# batch queue. Each slot covers both queued and running inference.
inference_slots = threading.BoundedSemaphore(MAX_IN_FLIGHT)
service_metrics = ServiceMetrics()
# One background thread owns the model and decodes queued requests in batches.
batcher = MicroBatcher(lambda: generator, MAX_BATCH_SIZE)

torch.set_num_threads(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator

    log_event(
        "model_loading",
        model="sshleifer/tiny-gpt2",
        pytorch_cpu_threads=torch.get_num_threads(),
        max_in_flight=MAX_IN_FLIGHT,
        max_batch_size=MAX_BATCH_SIZE,
    )
    try:
        text_pipeline = pipeline(
            "text-generation",
            model="sshleifer/tiny-gpt2",
            device="cpu",
        )
        generator = GreedyBatchGenerator(text_pipeline.model, text_pipeline.tokenizer)
    except Exception as exc:
        log_event("model_load_failed", error_type=type(exc).__name__)
        raise

    log_event("model_ready")
    try:
        yield
    finally:
        batcher.stop()
        generator = None
        log_event("model_unloaded")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    RequestObservabilityMiddleware,
    metrics=service_metrics,
    log_requests=request_logging_enabled(),
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=50, ge=1, le=256)


class GenerateResponse(BaseModel):
    generated_text: str
    max_new_tokens: int
    inference_time_ms: float
    lock_wait_ms: float
    batch_size: int


@app.get("/health")
async def health():
    """Lightweight liveness endpoint; does not test model readiness."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Only ready when a model pipeline has finished loading."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Process-local cumulative counters; protect this endpoint in deployments."""
    return {
        "max_in_flight": MAX_IN_FLIGHT,
        "max_batch_size": MAX_BATCH_SIZE,
        **service_metrics.snapshot(),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # This must run before queueing for the model; otherwise incoming requests
    # may silently queue there instead of receiving an overload reply.
    if not inference_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Inference capacity exceeded; retry later",
            headers={"Retry-After": "1"},
        )

    service_metrics.admitted()
    try:
        result = await asyncio.wrap_future(
            batcher.submit(request.prompt, request.max_new_tokens)
        )
        # lock_wait_ms keeps its name for benchmark compatibility: it is the
        # time spent queued before the model started this request's batch.
        service_metrics.record_inference(result.inference_time_ms, result.queue_wait_ms)
        return GenerateResponse(
            generated_text=result.generated_text,
            max_new_tokens=request.max_new_tokens,
            inference_time_ms=result.inference_time_ms,
            lock_wait_ms=result.queue_wait_ms,
            batch_size=result.batch_size,
        )
    finally:
        service_metrics.released()
        inference_slots.release()
