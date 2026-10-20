"""Batching and greedy-decoding checks with fake models (no model download)."""

import threading
import unittest
from types import SimpleNamespace

import torch

from batched_inference import GreedyBatchGenerator, MicroBatcher

EOS = 0
VOCAB = 10


class CountingModel:
    """Next token is (last token + 1) % VOCAB; records batch shapes."""

    def __init__(self):
        self.batch_shapes = []

    def eval(self):
        return self

    def __call__(self, input_ids, past_key_values, use_cache, logits_to_keep):
        self.batch_shapes.append(tuple(input_ids.shape))
        next_tokens = (input_ids[:, -1] + 1) % VOCAB
        logits = torch.zeros(input_ids.shape[0], 1, VOCAB)
        logits[torch.arange(input_ids.shape[0]), 0, next_tokens] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=object())


class DigitTokenizer:
    """Each character is a digit token; token 0 is EOS and decodes to ''."""

    eos_token_id = EOS

    def __call__(self, text):
        return {"input_ids": [int(char) for char in text]}

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(str(t) for t in tokens if not (skip_special_tokens and t == EOS))


class GreedyBatchGeneratorTests(unittest.TestCase):
    def test_respects_per_request_limits_eos_and_prompt_lengths(self):
        model = CountingModel()
        generate = GreedyBatchGenerator(model, DigitTokenizer())

        texts = generate(["1", "7", "12", "3"], [3, 5, 2, 1])

        # "7" -> 8, 9, then EOS stops it before its limit of 5.
        self.assertEqual(texts, ["1234", "789", "1234", "34"])
        # Equal-length prompts share batches; different lengths never pad.
        self.assertEqual(model.batch_shapes[0], (3, 1))
        self.assertIn((1, 2), model.batch_shapes)


class MicroBatcherTests(unittest.TestCase):
    def test_queued_requests_are_decoded_together(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def generator(prompts, max_new_tokens):
            calls.append(list(prompts))
            started.set()
            release.wait(timeout=5)
            return [prompt.upper() for prompt in prompts]

        batcher = MicroBatcher(lambda: generator, max_batch_size=2)
        try:
            first = batcher.submit("a", 1)
            self.assertTrue(started.wait(timeout=3))
            queued = [batcher.submit(prompt, 1) for prompt in ("b", "c", "d")]
            release.set()
            results = [future.result(timeout=3) for future in [first, *queued]]
        finally:
            batcher.stop()

        self.assertEqual(calls, [["a"], ["b", "c"], ["d"]])
        self.assertEqual([r.generated_text for r in results], ["A", "B", "C", "D"])
        self.assertEqual([r.batch_size for r in results], [1, 2, 2, 1])
        self.assertTrue(all(r.queue_wait_ms >= 0 for r in results))

    def test_failure_is_reported_to_whole_batch_and_batcher_recovers(self):
        fail = True

        def generator(prompts, max_new_tokens):
            if fail:
                raise RuntimeError("boom")
            return ["ok"] * len(prompts)

        batcher = MicroBatcher(lambda: generator, max_batch_size=4)
        try:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                batcher.submit("x", 1).result(timeout=3)
            fail = False
            self.assertEqual(batcher.submit("y", 1).result(timeout=3).generated_text, "ok")
        finally:
            batcher.stop()

    def test_batch_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            MicroBatcher(lambda: None, max_batch_size=0)


if __name__ == "__main__":
    unittest.main()
