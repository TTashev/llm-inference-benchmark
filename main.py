
import threading
import torch
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import pipeline


generator = None

inference_lock = threading.Lock()

torch.set_num_threads(1)

print("PyTorch CPU threads:", torch.get_num_threads())

class NewlineJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(content) + b"\n"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator

    print("Loading AI model...")

    generator = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",
        device="cpu"
    )

    print("Model ready!")

    yield

    generator = None


app = FastAPI(
    lifespan=lifespan,
    default_response_class=NewlineJSONResponse,
)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=50, ge=1, le=256)


class GenerateResponse(BaseModel):
    generated_text: str
    max_new_tokens: int
    inference_time_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):

    with inference_lock:
        start = time.perf_counter()

        result = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id
        )

        inference_time_ms = (
            time.perf_counter() - start
        ) * 1000

    return GenerateResponse(
        generated_text=result[0]["generated_text"],
        max_new_tokens=request.max_new_tokens,
        inference_time_ms=inference_time_ms
    )
