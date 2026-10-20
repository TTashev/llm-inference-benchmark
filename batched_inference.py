"""Dynamic micro-batching and batched greedy decoding for the tiny CPU model.

Requests are queued and a single background thread owns the model. Whenever
the model becomes free it takes every queued request (up to a batch limit) and
decodes them together, so concurrent requests share one forward pass per token
instead of waiting for the model one at a time.
"""

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np
import torch


class GreedyBatchGenerator:
    """Greedy decoding with a KV cache, producing the same text as the
    ``text-generation`` pipeline with ``do_sample=False`` for each prompt."""

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    @torch.inference_mode()
    def __call__(self, prompts: list[str], max_new_tokens: list[int]) -> list[str]:
        encoded = [self.tokenizer(prompt)["input_ids"] for prompt in prompts]
        texts: list[str | None] = [None] * len(prompts)
        # Batch prompts of equal token length together, so no padding or
        # attention masking is needed and each row matches unbatched output.
        by_length: dict[int, list[int]] = {}
        for index, ids in enumerate(encoded):
            by_length.setdefault(len(ids), []).append(index)
        for indices in by_length.values():
            new_tokens = self._decode(
                [encoded[i] for i in indices], [max_new_tokens[i] for i in indices]
            )
            for i, tokens in zip(indices, new_tokens):
                texts[i] = prompts[i] + self.tokenizer.decode(
                    tokens, skip_special_tokens=True
                )
        return texts

    def _decode(self, input_ids: list[list[int]], limits: list[int]) -> list[list[int]]:
        rows = len(input_ids)
        steps = max(limits)
        generated: list[list[int]] = [[] for _ in range(rows)]
        active = [True] * rows

        next_input = torch.tensor(input_ids, dtype=torch.long)
        past = None
        for step in range(steps):
            output = self.model(
                input_ids=next_input,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
            # numpy's argmax is several times faster than torch's on CPU for
            # 50k-wide vocabularies and keeps the same first-maximum tie rule.
            next_tokens = np.argmax(output.logits[:, -1, :].numpy(), axis=-1)

            for row, token in enumerate(next_tokens.tolist()):
                if not active[row]:
                    continue
                generated[row].append(token)
                if token == self.eos_token_id or len(generated[row]) >= limits[row]:
                    active[row] = False
            if not any(active) or step == steps - 1:
                break
            next_input = torch.from_numpy(next_tokens).unsqueeze(-1)
        return generated


@dataclass
class _Pending:
    prompt: str
    max_new_tokens: int
    future: Future
    enqueued_at: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class BatchResult:
    generated_text: str
    queue_wait_ms: float
    inference_time_ms: float
    batch_size: int


class MicroBatcher:
    """Single model-owner thread that drains the queue into batches."""

    def __init__(self, get_generator, max_batch_size: int):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self._get_generator = get_generator
        self.max_batch_size = max_batch_size
        self._queue: queue.SimpleQueue[_Pending | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def submit(self, prompt: str, max_new_tokens: int) -> Future:
        self._ensure_started()
        future: Future = Future()
        self._queue.put(_Pending(prompt, max_new_tokens, future))
        return future

    def stop(self, timeout: float | None = 5.0) -> None:
        with self._start_lock:
            thread, self._thread = self._thread, None
        if thread is not None:
            self._queue.put(None)
            thread.join(timeout)

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="inference-batcher", daemon=True
                )
                self._thread.start()

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            stop_after = False
            while len(batch) < self.max_batch_size:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stop_after = True
                    break
                batch.append(item)
            self._run_batch(batch)
            if stop_after:
                return

    def _run_batch(self, batch: list[_Pending]) -> None:
        batch = [item for item in batch if item.future.set_running_or_notify_cancel()]
        if not batch:
            return
        started = time.perf_counter()
        try:
            texts = self._get_generator()(
                [item.prompt for item in batch],
                [item.max_new_tokens for item in batch],
            )
            if len(texts) != len(batch):
                raise RuntimeError("Generator returned the wrong number of results")
        except BaseException as exc:
            for item in batch:
                item.future.set_exception(exc)
            if not isinstance(exc, Exception):
                raise
            return
        inference_ms = (time.perf_counter() - started) * 1000
        for item, text in zip(batch, texts):
            item.future.set_result(
                BatchResult(
                    generated_text=text,
                    queue_wait_ms=(started - item.enqueued_at) * 1000,
                    inference_time_ms=inference_ms,
                    batch_size=len(batch),
                )
            )
