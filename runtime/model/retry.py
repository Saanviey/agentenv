import logging, time
from runtime.types import Message, Response
from runtime.model.base import ModelClient, ModelUnavailable, ModelTimeout, ModelRateLimited

log = logging.getLogger("runtime.retry")
RETRYABLE = (ModelUnavailable, ModelTimeout, ModelRateLimited)  # temporary; anything else (bad key, 404) fails at once
MAX_DELAY_S = 30

# Wraps any client and retries temporary failures with exponential backoff (1s, 2s, 4s...).
# The loop never knows retries happened; it just sees a slower call, or the final error.
class RetryingClient(ModelClient):
    def __init__(self, inner: ModelClient, attempts: int = 3, base_delay_s: float = 1.0, sleep=time.sleep, on_retry=None):
        self.inner, self.attempts, self.base_delay_s, self.sleep = inner, attempts, base_delay_s, sleep
        self.on_retry = on_retry  # optional callback(attempt, delay_s, error), e.g. to trace retries

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Response:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.inner.complete(messages, tools)
            except RETRYABLE as e:
                limit = self.attempts * 2 if isinstance(e, ModelRateLimited) else self.attempts  # rate limits clear by waiting
                if attempt >= limit:
                    raise  # out of attempts: let the caller see the real error
                # never wait less than the backoff: a "retry in 1.4s" hint is often still too early when the minute's budget is full
                delay = max(getattr(e, "retry_after", None) or 0, self.base_delay_s * 2 ** (attempt - 1))
                delay = min(delay, MAX_DELAY_S)
                log.warning("model call failed: %s; retry %d/%d in %.1fs", str(e)[:120], attempt, limit - 1, delay)
                if self.on_retry:
                    self.on_retry(attempt, delay, e)
                self.sleep(delay)
