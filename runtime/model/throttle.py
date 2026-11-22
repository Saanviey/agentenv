import logging, time
from collections import deque
from runtime.types import Message, Response
from runtime.model.base import ModelClient
from runtime.context import estimate_tokens

log = logging.getLogger("runtime.throttle")
WINDOW_S = 60
OUTPUT_GUESS = 500  # providers count the reply too; we don't know its size before sending

# Wraps a client and WAITS before a call that would push the last minute's tokens over the provider's limit (B28).
# Proactive: avoids the 429 instead of retrying after it. Groq free tier: 8000 tokens per minute.
class ThrottledClient(ModelClient):
    def __init__(self, inner: ModelClient, tokens_per_minute: int, clock=time.monotonic, sleep=time.sleep, on_wait=None):
        self.inner, self.tpm, self.clock, self.sleep, self.on_wait = inner, tokens_per_minute, clock, sleep, on_wait
        self.sent: deque[tuple[float, int]] = deque()  # (when, tokens) for calls in the last minute

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Response:
        need = min(estimate_tokens(messages, tools) + OUTPUT_GUESS, self.tpm)  # a single call bigger than the limit can't wait its way in
        while (wait := self._wait_for(need)) > 0:
            log.info("throttle: %d tokens used in the last minute, need %d; waiting %.1fs", self._used(), need, wait)
            if self.on_wait:
                self.on_wait(wait, self._used(), need)
            self.sleep(wait)
        resp = self.inner.complete(messages, tools)
        self.sent.append((self.clock(), (resp.input_tokens + resp.output_tokens) or need))  # real count when the provider gives one
        return resp

    def _used(self) -> int:
        now = self.clock()
        while self.sent and now - self.sent[0][0] >= WINDOW_S:  # forget calls older than a minute
            self.sent.popleft()
        return sum(t for _, t in self.sent)

    # Out: seconds to wait until `need` more tokens fit in the window (0 = send now).
    def _wait_for(self, need: int) -> float:
        used = self._used()
        if used + need <= self.tpm:
            return 0
        for when, tokens in self.sent:  # wait until enough old calls drop out of the window
            used -= tokens
            if used + need <= self.tpm:
                return WINDOW_S - (self.clock() - when) + 0.1
        return WINDOW_S
