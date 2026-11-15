from abc import ABC, abstractmethod
from runtime.types import Message, Response

# Q3: backends raise OUR errors, so the loop never sees httpx / anthropic exceptions.
class ModelError(Exception): ...
class ModelUnavailable(ModelError): ...   # server down -> retryable
class ModelTimeout(ModelError): ...       # too slow    -> retryable
class ModelRateLimited(ModelError): ...   # HTTP 429, too many requests -> retry after waiting

# Interface every backend implements. ABC: can't be instantiated directly.
class ModelClient(ABC):
    # Q4: stateless. The caller passes the FULL history every time; the client stores nothing.
    # In: full history + tool schemas. Out: next reply (text and/or tool calls). Raises: ModelError.
    @abstractmethod  # subclasses must define this
    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Response: ...
