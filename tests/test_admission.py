"""Unit tests for bounded inference admission without loading a model."""

import asyncio
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import main


class SlowFakeGenerator:
    tokenizer = SimpleNamespace(eos_token_id=0)

    def __init__(self, started: threading.Event, release: threading.Event):
        self.started = started
        self.release = release

    def __call__(self, *args, **kwargs):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("Timed out waiting for test to release inference")
        return [{"generated_text": "fake completion"}]


class FailingFakeGenerator:
    tokenizer = SimpleNamespace(eos_token_id=0)

    def __call__(self, *args, **kwargs):
        raise RuntimeError("simulated inference failure")


class FastFakeGenerator:
    tokenizer = SimpleNamespace(eos_token_id=0)

    def __call__(self, *args, **kwargs):
        return [{"generated_text": "fake completion"}]


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.request = main.GenerateRequest(prompt="test", max_new_tokens=1)

    def test_rejects_at_capacity_then_recovers(self):
        started = threading.Event()
        release = threading.Event()

        async def scenario():
            task = asyncio.create_task(main.generate(self.request))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                with self.assertRaises(HTTPException) as caught:
                    await main.generate(self.request)
                self.assertEqual(caught.exception.status_code, 503)
                self.assertEqual(caught.exception.headers["Retry-After"], "1")
            finally:
                release.set()
                await task

            # A released permit must be reusable after the first completion.
            response = await main.generate(self.request)
            self.assertEqual(response.generated_text, "fake completion")

        with (
            patch.object(main, "generator", SlowFakeGenerator(started, release)),
            patch.object(main, "inference_slots", threading.BoundedSemaphore(1)),
        ):
            asyncio.run(scenario())

    def test_releases_slot_after_inference_exception(self):
        async def scenario():
            with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
                await main.generate(self.request)
            with patch.object(main, "generator", FastFakeGenerator()):
                response = await main.generate(self.request)
                self.assertEqual(response.generated_text, "fake completion")

        with (
            patch.object(main, "generator", FailingFakeGenerator()),
            patch.object(main, "inference_slots", threading.BoundedSemaphore(1)),
        ):
            asyncio.run(scenario())

    def test_admits_two_and_rejects_third(self):
        entered = 0
        both_admitted = asyncio.Event()
        release = asyncio.Event()

        async def fake_threadpool(_fn, _request):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_admitted.set()
            await release.wait()
            return main.GenerateResponse(
                generated_text="fake",
                max_new_tokens=1,
                inference_time_ms=0,
                lock_wait_ms=0,
            )

        async def scenario():
            first = asyncio.create_task(main.generate(self.request))
            second = asyncio.create_task(main.generate(self.request))
            try:
                await asyncio.wait_for(both_admitted.wait(), timeout=3)
                with self.assertRaises(HTTPException) as caught:
                    await main.generate(self.request)
                self.assertEqual(caught.exception.status_code, 503)
            finally:
                release.set()
                await asyncio.gather(first, second)

            response = await main.generate(self.request)
            self.assertEqual(response.generated_text, "fake")

        with (
            patch.object(main, "inference_slots", threading.BoundedSemaphore(2)),
            patch.object(main, "run_in_threadpool", fake_threadpool),
        ):
            asyncio.run(scenario())

    def test_capacity_must_be_positive(self):
        for invalid in ("0", "-1", "not-a-number"):
            with self.subTest(invalid=invalid):
                with patch.dict(os.environ, {"MAX_IN_FLIGHT": invalid}):
                    with self.assertRaises(ValueError):
                        main.read_max_in_flight()


if __name__ == "__main__":
    unittest.main()
