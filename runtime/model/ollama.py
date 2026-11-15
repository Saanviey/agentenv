import time, httpx
from runtime.types import Message, Response, ToolCall
from runtime.model.base import ModelClient, ModelError, ModelUnavailable, ModelTimeout
from runtime.model.fallback import parse_text_tool_call

# Adapter: our Messages -> Ollama /api/chat -> our Response.
class OllamaClient(ModelClient):
    def __init__(self, name: str, url: str, timeout_s: float):
        self.name = name
        self.http = httpx.Client(base_url=url, timeout=timeout_s)  # reusable connection; paths join onto base_url

    # In: full history + tool schemas. Out: Response. Raises only our ModelErrors.
    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Response:
        body = {
            "model": self.name,
            "messages": [_to_ollama(m) for m in messages],
            "stream": False,  # one JSON reply, not token-by-token chunks
        }
        if tools:
            body["tools"] = tools  # Ollama pastes these into the prompt via the model's template
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
        msg = data["message"]
        content = msg.get("content", "")
        calls = [ToolCall(c["function"]["name"], c["function"].get("arguments") or {}, c.get("id"))
                 for c in msg.get("tool_calls") or []]  # Ollama parsed these from the model's <tool_call> tags
        if not calls and tools:
            call = parse_text_tool_call(content, {t["function"]["name"] for t in tools})
            if call:
                calls, content = [call], ""  # history then shows a proper tool call, nudging the next turn
        if calls:
            stop = "tool_use"
        else:
            stop = "length" if data.get("done_reason") == "length" else "end"
        return Response(
            message=Message("assistant", content, tool_calls=calls),
            stop_reason=stop,
            input_tokens=data.get("prompt_eval_count", 0),  # Ollama's name for input tokens
            output_tokens=data.get("eval_count", 0),        # Ollama's name for output tokens
            latency_ms=(time.perf_counter() - start) * 1000,
            model=data.get("model", self.name),
            raw=data,
        )

# Our Message -> Ollama's message dict.
def _to_ollama(m: Message) -> dict:
    d = {"role": m.role, "content": m.content}
    if m.tool_calls:  # assistant turn that requested tools
        d["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
    if m.role == "tool" and m.tool_name:
        d["tool_name"] = m.tool_name  # which call this result answers
    return d
