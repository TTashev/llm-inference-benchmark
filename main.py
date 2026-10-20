"""Tiny CPU inference service with admission control and observability."""

import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from observability import (
    RequestObservabilityMiddleware,
    ServiceMetrics,
    log_event,
    request_logging_enabled,
)
from pydantic import BaseModel, Field
from transformers import pipeline


def read_max_in_flight() -> int:
    """Maximum admitted requests, including active inference and lock waiters."""
    raw = os.getenv("MAX_IN_FLIGHT", "4")
    try:
        capacity = int(raw)
    except ValueError as exc:
        raise ValueError("MAX_IN_FLIGHT must be a positive integer") from exc
    if capacity < 1:
        raise ValueError("MAX_IN_FLIGHT must be a positive integer")
    return capacity


MAX_IN_FLIGHT = read_max_in_flight()
generator = None
inference_lock = threading.Lock()
# Non-blocking admission at the async HTTP handler, before work reaches the
# worker thread pool. Each slot covers both inference and waiting for the lock.
inference_slots = threading.BoundedSemaphore(MAX_IN_FLIGHT)
service_metrics = ServiceMetrics()

torch.set_num_threads(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator

    log_event(
        "model_loading",
        model="sshleifer/tiny-gpt2",
        pytorch_cpu_threads=torch.get_num_threads(),
        max_in_flight=MAX_IN_FLIGHT,
    )
    try:
        generator = pipeline(
            "text-generation",
            model="sshleifer/tiny-gpt2",
            device="cpu",
        )
    except Exception as exc:
        log_event("model_load_failed", error_type=type(exc).__name__)
        raise

    log_event("model_ready")
    try:
        yield
    finally:
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
    return {"max_in_flight": MAX_IN_FLIGHT, **service_metrics.snapshot()}


def run_inference(request: GenerateRequest) -> GenerateResponse:
    """Run the blocking model call in a worker; one call owns the model lock."""
    start_wait = time.perf_counter()

    with inference_lock:
        lock_wait_ms = (time.perf_counter() - start_wait) * 1000
        start_inference = time.perf_counter()

        result = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        inference_time_ms = (time.perf_counter() - start_inference) * 1000

    service_metrics.record_inference(inference_time_ms, lock_wait_ms)
    return GenerateResponse(
        generated_text=result[0]["generated_text"],
        max_new_tokens=request.max_new_tokens,
        inference_time_ms=inference_time_ms,
        lock_wait_ms=lock_wait_ms,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # This must run before offloading to the thread pool; otherwise incoming
    # requests may silently queue there instead of receiving an overload reply.
    if not inference_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Inference capacity exceeded; retry later",
            headers={"Retry-After": "1"},
        )

    service_metrics.admitted()
    try:
        return await run_in_threadpool(run_inference, request)
    finally:
        service_metrics.released()
        inference_slots.release()
