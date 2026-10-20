"""Unit tests for bounded inference admission without loading a model."""

import asyncio
import os
import threading
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import main


class SlowFakeGenerator:
    def __init__(self, started: threading.Event, release: threading.Event):
        self.started = started
        self.release = release

    def __call__(self, prompts, max_new_tokens):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("Timed out waiting for test to release inference")
        return ["fake completion"] * len(prompts)


class FailingFakeGenerator:
    def __call__(self, prompts, max_new_tokens):
        raise RuntimeError("simulated inference failure")


class FastFakeGenerator:
    def __call__(self, prompts, max_new_tokens):
        return ["fake completion"] * len(prompts)


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
        started = threading.Event()
        release = threading.Event()

        async def scenario():
            first = asyncio.create_task(main.generate(self.request))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                # The second request is admitted and queued behind the first batch.
                second = asyncio.create_task(main.generate(self.request))
                await asyncio.sleep(0)
                with self.assertRaises(HTTPException) as caught:
                    await main.generate(self.request)
                self.assertEqual(caught.exception.status_code, 503)
            finally:
                release.set()
            responses = await asyncio.gather(first, second)
            self.assertEqual(
                [response.generated_text for response in responses],
                ["fake completion", "fake completion"],
            )

            response = await main.generate(self.request)
            self.assertEqual(response.generated_text, "fake completion")

        with (
            patch.object(main, "generator", SlowFakeGenerator(started, release)),
            patch.object(main, "inference_slots", threading.BoundedSemaphore(2)),
        ):
            asyncio.run(scenario())

    def test_batch_size_must_be_positive(self):
        for invalid in ("0", "-1", "not-a-number"):
            with self.subTest(invalid=invalid):
                with patch.dict(os.environ, {"MAX_BATCH_SIZE": invalid}):
                    with self.assertRaises(ValueError):
                        main.read_max_batch_size()

    def test_capacity_must_be_positive(self):
        for invalid in ("0", "-1", "not-a-number"):
            with self.subTest(invalid=invalid):
                with patch.dict(os.environ, {"MAX_IN_FLIGHT": invalid}):
                    with self.assertRaises(ValueError):
                        main.read_max_in_flight()


if __name__ == "__main__":
    unittest.main()
