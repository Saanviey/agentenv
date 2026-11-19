import unittest
from runtime.model.base import ModelError, ModelUnavailable, ModelTimeout, ModelRateLimited
from runtime.model.retry import RetryingClient
from tests.fakes import FakeClient, text

def client(script, attempts=3):
    waits = []
    return RetryingClient(FakeClient(script), attempts=attempts, base_delay_s=1, sleep=waits.append), waits

class Retry(unittest.TestCase):
    def test_recovers_from_temporary_failures(self):
        c, waits = client([ModelUnavailable("down"), ModelTimeout("slow"), text("hi")])
        self.assertEqual(c.complete([]).message.content, "hi")
        self.assertEqual(waits, [1, 2])  # exponential backoff

    def test_gives_up_after_attempts(self):
        c, waits = client([ModelUnavailable("down")] * 3)
        with self.assertRaises(ModelUnavailable):
            c.complete([])
        self.assertEqual(len(waits), 2)

    def test_does_not_retry_permanent_errors(self):
        c, waits = client([ModelError("401 bad key"), text("never")])
        with self.assertRaises(ModelError):
            c.complete([])
        self.assertEqual(waits, [])

    def test_uses_server_retry_after(self):
        c, waits = client([ModelRateLimited("429", retry_after=7), text("hi")])
        c.complete([])
        self.assertEqual(waits, [7])

    def test_rate_limits_get_more_attempts(self):
        c, waits = client([ModelRateLimited("429")] * 5 + [text("hi")])
        self.assertEqual(c.complete([]).message.content, "hi")
        self.assertEqual(len(waits), 5)

    def test_short_retry_after_does_not_undercut_backoff(self):
        c, waits = client([ModelRateLimited("429", retry_after=0.5)] * 3 + [text("hi")])
        c.complete([])
        self.assertEqual(waits, [1, 2, 4])

    def test_caps_long_waits(self):
        c, waits = client([ModelRateLimited("429", retry_after=600), text("hi")])
        c.complete([])
        self.assertEqual(waits, [30])

if __name__ == "__main__":
    unittest.main()
