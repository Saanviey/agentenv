from abc import ABC, abstractmethod
from runtime.types import Message, Response

# Q3: backends raise OUR errors, so the loop never sees httpx / anthropic exceptions.
class ModelError(Exception): ...
class ModelUnavailable(ModelError): ...   # server down -> retryable
class ModelTimeout(ModelError): ...       # too slow    -> retryable
class ModelRateLimited(ModelError):       # HTTP 429, too many requests -> retry after waiting
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after       # seconds the server asked us to wait, if it said
class ModelBadOutput(ModelError):          # the model produced a broken tool call the server refused -> tell the model, let it retry
    def __init__(self, msg: str, generation: str = ""):
        super().__init__(msg)
        self.generation = generation          # what the model actually wrote, if the server shared it

# Interface every backend implements. ABC: can't be instantiated directly.
class ModelClient(ABC):
    # Q4: stateless. The caller passes the FULL history every time; the client stores nothing.
    # In: full history + tool schemas. Out: next reply (text and/or tool calls). Raises: ModelError.
    @abstractmethod  # subclasses must define this
    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Response: ...
