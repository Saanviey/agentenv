import json, time, uuid, httpx
from runtime.types import Message, Response, ToolCall
from runtime.model.base import ModelClient, ModelError, ModelUnavailable, ModelTimeout, ModelRateLimited, ModelBadOutput
from runtime.model.fallback import parse_text_tool_call

# Adapter for any OpenAI-compatible chat API: Groq, OpenRouter, Cerebras, Ollama's /v1, OpenAI itself.
class OpenAICompatClient(ModelClient):
    def __init__(self, name: str, url: str, timeout_s: float, api_key: str | None = None):
        self.name = name
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=url, timeout=timeout_s, headers=headers)  # headers sent on every request

    # In: full history + tool schemas. Out: Response. Raises only our ModelErrors.
    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Response:
        body = {"model": self.name, "messages": [_to_openai(m) for m in messages]}
        if tools:
            body["tools"] = tools  # same schema format ToolRegistry.schemas() already produces
        start = time.perf_counter()
        try:
            r = self.http.post("/chat/completions", json=body)
            r.raise_for_status()
        except httpx.ConnectError as e:
            raise ModelUnavailable(f"API not reachable: {e}") from e
        except httpx.TimeoutException as e:
            raise ModelTimeout(str(e)) from e
        except httpx.HTTPStatusError as e:
            code, text = e.response.status_code, e.response.text[:500]
            if code == 429:
                wait = e.response.headers.get("retry-after")  # server's hint, in seconds
                raise ModelRateLimited(f"429 rate limited: {text}",
                                       float(wait) if wait and wait.replace(".", "", 1).isdigit() else None) from e
            if code == 400 and "tool_use_failed" in text:  # Groq checks tool-call JSON server-side and refuses bad ones
                try:
                    gen = e.response.json()["error"].get("failed_generation", "")
                except Exception:
                    gen = ""
                raise ModelBadOutput(f"malformed tool call: {text}", gen) from e
            if code >= 500:
                raise ModelUnavailable(f"{code} server error: {text}") from e
            raise ModelError(f"{code}: {text}") from e  # 401 bad key, 400 bad request, 404 unknown model

        data = r.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""  # null when the reply is only tool calls
        calls = [ToolCall(c["function"]["name"], _parse_args(c["function"].get("arguments")), c.get("id"))
                 for c in msg.get("tool_calls") or []]
        if not calls and tools:  # weak models may write the call as text instead
            call = parse_text_tool_call(content, {t["function"]["name"] for t in tools})
            if call:
                calls, content = [call], ""
        usage = data.get("usage") or {}
        return Response(
            message=Message("assistant", content, tool_calls=calls),
            stop_reason="tool_use" if calls else ("length" if choice.get("finish_reason") == "length" else "end"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
            model=data.get("model", self.name),
            raw=data,
        )

# OpenAI sends arguments as a JSON *string*; turn it into a dict, or leave the string for tools.run() to report.
def _parse_args(a):
    if isinstance(a, str):
        try:
            return json.loads(a) if a.strip() else {}
        except json.JSONDecodeError:
            return a
    return a or {}

# Our Message -> OpenAI's message dict.
def _to_openai(m: Message) -> dict:
    d = {"role": m.role, "content": m.content}
    if m.tool_calls:
        for c in m.tool_calls:
            c.id = c.id or f"call_{uuid.uuid4().hex[:8]}"  # OpenAI requires an id on every call
        d["tool_calls"] = [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                           for c in m.tool_calls]
    if m.role == "tool":
        d["tool_call_id"] = m.tool_call_id  # must match the id of the call it answers
    return d
