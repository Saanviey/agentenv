from abc import ABC, abstractmethod
from runtime.types import Message, Response

# Q3: backends raise OUR errors, so the loop never sees httpx / anthropic exceptions.
class ModelError(Exception): ...
class ModelUnavailable(ModelError): ...   # server down -> retryable
class ModelTimeout(ModelError): ...       # too slow    -> retryable

# Interface every backend implements. ABC: can't be instantiated directly.
class ModelClient(ABC):
    # Q4: stateless. The caller passes the FULL history every time; the client stores nothing.
    # In: full history. Out: next reply. Raises: ModelError.
    @abstractmethod  # subclasses must define this
    def complete(self, messages: list[Message]) -> Response: ...
