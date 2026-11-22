import unittest
from runtime.model.throttle import ThrottledClient, OUTPUT_GUESS
from runtime.types import Message
from tests.fakes import FakeClient, text

class Clock:
    def __init__(self):
        self.now, self.waits = 0.0, []
    def __call__(self):
        return self.now
    def sleep(self, s):
        self.waits.append(round(s, 1))
        self.now += s

def reply(tokens):
    r = text("ok")
    r.input_tokens, r.output_tokens = tokens, 0
    return r

class Throttle(unittest.TestCase):
    def make(self, replies, tpm=8000):
        clock = Clock()
        return ThrottledClient(FakeClient(replies), tpm, clock=clock, sleep=clock.sleep), clock

    def test_under_limit_no_wait(self):
        c, clock = self.make([reply(2000)] * 3)
        for _ in range(3):
            c.complete([Message("user", "hi")])
        self.assertEqual(clock.waits, [])

    def test_waits_until_old_calls_leave_the_window(self):
        c, clock = self.make([reply(5000), reply(5000)])
        c.complete([Message("user", "hi")])
        clock.now = 10
        c.complete([Message("user", "x" * 16000)])   # ~4000 + 500 tokens: 5000 + 4500 > 8000
        self.assertEqual(clock.waits, [50.1])  # first call drops out at t=60

    def test_only_needed_calls_are_waited_out(self):
        c, clock = self.make([reply(3000), reply(3000), reply(1000)])
        c.complete([Message("user", "a")])
        clock.now = 20
        c.complete([Message("user", "b")])
        clock.now = 30
        c.complete([Message("user", "c")])   # 6000 used + ~504 needed fits: no wait
        self.assertEqual(clock.waits, [])

    def test_request_bigger_than_limit_is_capped_not_stuck(self):
        c, clock = self.make([reply(100)], tpm=1000)
        c.complete([Message("user", "x" * 20000)])
        self.assertEqual(clock.waits, [])

    def test_callback(self):
        seen = []
        clock = Clock()
        c = ThrottledClient(FakeClient([reply(7900), reply(10)]), 8000, clock=clock, sleep=clock.sleep,
                            on_wait=lambda w, used, need: seen.append((used, need)))
        c.complete([Message("user", "a")])
        c.complete([Message("user", "b")])
        self.assertEqual(seen[0][0], 7900)
        self.assertGreater(seen[0][1], OUTPUT_GUESS)

class FactoryOrder(unittest.TestCase):
    def test_retry_wraps_throttle_wraps_backend(self):
        from runtime.model.factory import build_client
        c = build_client({"provider": "ollama", "name": "m", "url": "http://x", "timeout_s": 1, "tokens_per_minute": 8000})
        self.assertEqual((type(c).__name__, type(c.inner).__name__, type(c.inner.inner).__name__),
                         ("RetryingClient", "ThrottledClient", "OllamaClient"))

if __name__ == "__main__":
    unittest.main()
