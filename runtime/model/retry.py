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
        for attempt in range(1, self.attempts + 1):
            try:
                return self.inner.complete(messages, tools)
            except RETRYABLE as e:
                if attempt == self.attempts:
                    raise  # out of attempts: let the caller see the real error
                delay = getattr(e, "retry_after", None) or self.base_delay_s * 2 ** (attempt - 1)
                delay = min(delay, MAX_DELAY_S)
                log.warning("model call failed: %s; retry %d/%d in %.1fs", e, attempt, self.attempts - 1, delay)
                if self.on_retry:
                    self.on_retry(attempt, delay, e)
                self.sleep(delay)
