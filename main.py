import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline


generator = None
inference_lock = threading.Lock()

torch.set_num_threads(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator

    print("PyTorch CPU threads:", torch.get_num_threads())
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


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
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
