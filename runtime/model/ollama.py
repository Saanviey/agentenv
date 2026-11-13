import time, httpx
from runtime.types import Message, Response
from runtime.model.base import ModelClient, ModelError, ModelUnavailable, ModelTimeout

# Adapter: our Messages -> Ollama /api/chat -> our Response.
class OllamaClient(ModelClient):
    def __init__(self, name: str, url: str, timeout_s: float):
        self.name = name
        self.http = httpx.Client(base_url=url, timeout=timeout_s)  # reusable connection; paths join onto base_url

    # In: full history. Out: Response. Raises only our ModelErrors.
    def complete(self, messages: list[Message]) -> Response:
        body = {
            "model": self.name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,  # one JSON reply, not token-by-token chunks
        }
        start = time.perf_counter()
        try:
            r = self.http.post("/api/chat", json=body)  # json= serializes body and sets the header
            r.raise_for_status()  # httpx doesn't raise on 4xx/5xx unless asked
        except httpx.ConnectError as e:       # nothing listening on the port
            raise ModelUnavailable(f"Ollama not reachable: {e}") from e
        except httpx.TimeoutException as e:   # exceeded timeout_s
            raise ModelTimeout(str(e)) from e
        except httpx.HTTPStatusError as e:    # server replied with an error, e.g. 404
            raise ModelError(f"{e.response.status_code}: {e.response.text}") from e

        data = r.json()
        return Response(
            message=Message("assistant", data["message"]["content"]),
            stop_reason="length" if data.get("done_reason") == "length" else "end",
            input_tokens=data.get("prompt_eval_count", 0),  # Ollama's name for input tokens
            output_tokens=data.get("eval_count", 0),        # Ollama's name for output tokens
            latency_ms=(time.perf_counter() - start) * 1000,
            model=data.get("model", self.name),
            raw=data,
        )
