"""Tiny CPU inference service with bounded request admission."""

import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
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

torch.set_num_threads(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator

    print("PyTorch CPU threads:", torch.get_num_threads())
    print("Maximum admitted inference requests:", MAX_IN_FLIGHT)
    print("Loading AI model...")

    generator = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",
        device="cpu",
    )

    print("Model ready!")
    yield
    generator = None


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=50, ge=1, le=256)


class GenerateResponse(BaseModel):
    generated_text: str
    max_new_tokens: int
    inference_time_ms: float
    lock_wait_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


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

    try:
        return await run_in_threadpool(run_inference, request)
    finally:
        inference_slots.release()
