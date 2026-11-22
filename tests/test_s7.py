import json, tempfile, unittest
from pathlib import Path
import httpx
from runtime.context import ContextManager, ContextOverflow, estimate_tokens, _turns
from runtime.loop import run_agent
from runtime.model.base import ModelContextOverflow
from runtime.model.ollama import OllamaClient
from runtime.model.openai_compat import OpenAICompatClient
from runtime.policy import approve_all
from runtime.tools import default_tools
from runtime.trace import Tracer, load
from runtime.types import Message, ToolCall
from tests.fakes import FakeClient, call

BIG = "x = 1  # filler line for a big file\n" * 120   # ~4300 chars

def conversation(n_turns: int, result: str = BIG) -> list[Message]:
    msgs = [Message("system", "rules"), Message("user", "task")]
    for i in range(n_turns):
        msgs.append(Message("assistant", "", tool_calls=[ToolCall("read_file", {"path": f"f{i}.py"}, f"c{i}")]))
        msgs.append(Message("tool", result, tool_call_id=f"c{i}", tool_name="read_file"))
    return msgs

def well_formed(msgs: list[Message]) -> bool:  # every tool result directly follows the assistant call(s) it answers
    for i, m in enumerate(msgs):
        if m.role == "tool":
            j = i - 1
            while msgs[j].role == "tool":
                j -= 1
            if msgs[j].role != "assistant" or m.tool_call_id not in [c.id for c in msgs[j].tool_calls]:
                return False
    return True

class Estimate(unittest.TestCase):
    def test_grows_with_content_and_tools(self):
        a = estimate_tokens(conversation(1))
        self.assertGreater(estimate_tokens(conversation(2)), a)
        self.assertGreater(estimate_tokens(conversation(1), [{"type": "function", "function": {"name": "x" * 400}}]), a)

    def test_calibration(self):
        cm = ContextManager(1000)
        cm.observe(100, 150)
        self.assertAlmostEqual(cm.scale, 1.25)
        cm.observe(100, 30)          # looks like a cached count: ignored
        self.assertAlmostEqual(cm.scale, 1.25)
        for _ in range(20):
            cm.observe(100, 1000)    # way off: clamped
        self.assertEqual(cm.scale, 2.0)

class Fit(unittest.TestCase):
    def test_under_budget_untouched(self):
        msgs = conversation(2)
        self.assertIsNone(ContextManager(100_000).fit(msgs))
        self.assertEqual(msgs[3].content, BIG)

    def test_elides_oldest_first_keeps_pinned_and_recent(self):
        msgs = conversation(6)
        c = ContextManager(estimate_tokens(msgs) - 500, keep_recent=3).fit(msgs)
        self.assertEqual(c.method, "elide")
        self.assertEqual(msgs[3].tag, "elided")                      # oldest result
        self.assertTrue(all(m.content == BIG for m in msgs[-6:] if m.role == "tool"))  # last 3 turns whole
        self.assertEqual([m.content for m in msgs[:2]], ["rules", "task"])
        self.assertTrue(well_formed(msgs))

    def test_summarizes_when_eliding_is_not_enough(self):
        msgs = conversation(8)
        seen = []
        cm = ContextManager(1300, summarize=lambda text: seen.append(text) or "read f0-f4; bug not found yet", keep_recent=3)
        c = cm.fit(msgs)
        self.assertIn("summarize(model)", c.method)
        self.assertEqual(msgs[2].tag, "summary")
        self.assertIn("bug not found yet", msgs[2].content)
        self.assertIn("read_file", seen[0])
        self.assertTrue(well_formed(msgs))
        self.assertLessEqual(c.after, 1300)

    def test_summarizer_failure_falls_back_to_mechanical(self):
        def broken(text):
            raise RuntimeError("model down")
        msgs = conversation(8)
        c = ContextManager(1800, summarize=broken).fit(msgs)
        self.assertIn("summarize(mechanical)", c.method)
        self.assertIn("called read_file", msgs[2].content)

    def test_second_compaction_folds_earlier_summary(self):
        msgs = conversation(8)
        cm = ContextManager(1800)
        cm.fit(msgs)
        first = msgs[2].content
        msgs += conversation(6)[2:]
        cm.fit(msgs)
        self.assertEqual(sum(m.tag == "summary" for m in msgs), 1)
        self.assertIn(first.split("\n", 2)[2][:40], msgs[2].content)

    def test_last_turn_never_touched(self):
        msgs = conversation(5)
        ContextManager(1800, keep_recent=1).fit(msgs)
        self.assertEqual(msgs[-1].content, BIG)

    def test_overflow_when_pinned_alone_too_big(self):
        msgs = [Message("system", "rules"), Message("user", "y" * 20_000)]
        with self.assertRaises(ContextOverflow):
            ContextManager(1000).fit(msgs)

    def test_turns_start_at_assistant(self):
        msgs = conversation(2) + [Message("user", "reminder")]
        self.assertEqual(_turns(msgs), [(2, 4), (4, 7)])

class Workspace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "big.py").write_text("".join(f"line_{i} = {i}\n" for i in range(1, 801)))   # ~10k chars
        (self.root / "small.py").write_text("def target():\n    return 42\n")
        (self.root / ".venv").mkdir()
        (self.root / ".venv" / "lib.py").write_text("def target(): pass\n")
        self.tools = default_tools(self.root)

    def tearDown(self):
        self._tmp.cleanup()

class Offload(Workspace):
    def test_big_output_is_offloaded_with_preview(self):
        out = self.tools.run("read_file", {"path": "big.py"})
        self.assertIn("line_1 = 1", out)
        self.assertIn("line_800 = 800", out)                 # tail shown
        self.assertNotIn("line_400 = 400", out)              # middle not
        self.assertIn("read_output(id=1", out)
        self.assertEqual(len(self.tools.session.outputs), 1)
        self.assertLess(len(out), 3200)

    def test_read_output_pages_to_the_end(self):
        self.tools.run("read_file", {"path": "big.py"})
        full, offset, got = self.tools.session.outputs[0], 0, ""
        while True:
            page = self.tools.run("read_output", {"id": 1, "offset": offset})
            header, body = page.split("\n", 1)
            got += body
            if "(end)" in header:
                break
            offset = int(header.split("Next: read_output(id=1, offset=")[1].split(")")[0])
        self.assertEqual(got, full)

    def test_read_output_bad_id(self):
        self.assertIn("no output #3", self.tools.run("read_output", {"id": 3}))

    def test_small_output_not_offloaded(self):
        self.assertEqual(self.tools.run("read_file", {"path": "small.py"}), "def target():\n    return 42\n")

class Ranges(Workspace):
    def test_line_range(self):
        out = self.tools.run("read_file", {"path": "big.py", "start_line": 400, "end_line": 402})
        self.assertEqual(out, "[big.py lines 400-402 of 800]\nline_400 = 400\nline_401 = 401\nline_402 = 402\n")

    def test_range_past_end_is_clamped(self):
        self.assertIn("lines 799-800 of 800", self.tools.run("read_file", {"path": "big.py", "start_line": 799, "end_line": 5000}))

    def test_bad_range(self):
        self.assertIn("has 800 lines", self.tools.run("read_file", {"path": "big.py", "start_line": 900}))

class Search(Workspace):
    def test_finds_lines_and_skips_venv(self):
        out = self.tools.run("search", {"pattern": r"def target"})
        self.assertEqual(out, "small.py:1: def target():")

    def test_invalid_regex_is_literal(self):
        self.assertEqual(self.tools.run("search", {"pattern": "target("}), "small.py:1: def target():")  # "(" alone is bad regex
        self.assertIn("no matches for 'target(x'", self.tools.run("search", {"pattern": "target(x"}))

    def test_caps_matches(self):
        out = self.tools.run("search", {"pattern": "line_", "path": "big.py"})
        self.assertIn("stopped at 50 matches", out)

    def test_search_output_counts_as_seen(self):
        self.tools.run("search", {"pattern": "target"})
        self.assertIn("small.py", self.tools.session.seen[0])

class LoopIntegration(Workspace):
    def test_long_run_stays_under_budget(self):
        path = self.root / "trace.jsonl"
        tr = Tracer(path)
        reads = [call("read_file", {"path": "big.py", "start_line": i * 100 + 1, "end_line": i * 100 + 100}) for i in range(8)]
        r = run_agent(FakeClient(reads + [call("finish", {"answer": "done"})]), self.tools,
                      [Message("system", "s"), Message("user", "t")], 20, approve=approve_all, trace=tr,
                      context=ContextManager(2500, keep_recent=2))
        tr.close()
        ev = load(path)
        self.assertEqual(r.stopped, "finish")
        self.assertTrue(any(e["event"] == "compaction" for e in ev))
        self.assertTrue(all(e["est_tokens"] <= 2500 for e in ev if e["event"] == "model_request"))
        last = [e for e in ev if e["event"] == "model_request"][-1]
        self.assertEqual([m["role"] for m in last["new_messages"]], ["assistant", "tool"])  # trace bookkeeping survives compaction

    def test_413_lowers_budget_compacts_and_retries(self):
        reads = [call("read_file", {"path": "big.py", "start_line": i * 100 + 1, "end_line": i * 100 + 100}) for i in range(4)]
        cm = ContextManager(100_000, keep_recent=1)
        client = FakeClient(reads + [ModelContextOverflow("413 too large"), call("finish", {"answer": "done"})])
        r = run_agent(client, self.tools, [Message("system", "s"), Message("user", "t")], 20, approve=approve_all, context=cm)
        self.assertEqual(r.stopped, "finish")
        self.assertLess(cm.budget, 100_000)
        self.assertTrue(any(m.tag in ("elided", "summary") for m in r.messages))

    def test_413_without_context_manager_still_raises(self):
        with self.assertRaises(ModelContextOverflow):
            run_agent(FakeClient([ModelContextOverflow("413")]), self.tools, [Message("system", "s"), Message("user", "t")])

class Backends(unittest.TestCase):
    def test_ollama_sends_num_ctx(self):
        sent = {}
        def handler(req):
            sent.update(json.loads(req.content))
            return httpx.Response(200, json={"message": {"content": "hi"}, "done_reason": "stop"})
        OllamaClient("m", "http://x", 5, num_ctx=8192, transport=httpx.MockTransport(handler)).complete([Message("user", "hi")])
        self.assertEqual(sent["options"], {"num_ctx": 8192})

    def test_413_becomes_context_overflow(self):
        def handler(req):
            return httpx.Response(413, json={"error": {"message": "Request too large for model in organization org_ABC123"}})
        c = OpenAICompatClient("m", "http://x", 5, transport=httpx.MockTransport(handler))
        with self.assertRaises(ModelContextOverflow) as e:
            c.complete([Message("user", "hi")])
        self.assertNotIn("org_ABC123", str(e.exception))

if __name__ == "__main__":
    unittest.main()

class NetworkErrors(unittest.TestCase):  # B22: every httpx network failure must become our retryable error
    def test_dropped_connection_is_model_unavailable(self):
        from runtime.model.base import ModelUnavailable
        def handler(req):
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        for c in (OpenAICompatClient("m", "http://x", 5, transport=httpx.MockTransport(handler)),
                  OllamaClient("m", "http://x", 5, transport=httpx.MockTransport(handler))):
            with self.assertRaises(ModelUnavailable):
                c.complete([Message("user", "hi")])

    def test_timeout_still_model_timeout(self):
        from runtime.model.base import ModelTimeout
        def handler(req):
            raise httpx.ReadTimeout("slow")
        with self.assertRaises(ModelTimeout):
            OpenAICompatClient("m", "http://x", 5, transport=httpx.MockTransport(handler)).complete([Message("user", "hi")])

class SummaryTrust(unittest.TestCase):  # B25: summaries are model output; core facts come from the runtime
    def test_facts_block_under_summary(self):
        msgs = conversation(8)
        ContextManager(1800, summarize=lambda t: "looked at files").fit(msgs, facts="- files read: f0.py")
        self.assertIn("Recorded by the runtime (reliable):\n- files read: f0.py", msgs[2].content)

    def test_old_facts_not_resummarized(self):
        msgs = conversation(8)
        seen = []
        cm = ContextManager(1800, summarize=lambda t: seen.append(t) or "s")
        cm.fit(msgs, facts="- files read: OLD.py")
        msgs += conversation(6)[2:]
        cm.fit(msgs, facts="- files read: NEW.py")
        self.assertNotIn("OLD.py", seen[1])
        self.assertIn("NEW.py", msgs[2].content)
        self.assertNotIn("OLD.py", msgs[2].content)

    def test_prompt_forbids_invented_code(self):
        from runtime.context import SUMMARY_PROMPT
        self.assertIn("Never reconstruct", SUMMARY_PROMPT)

    def test_session_facts(self):
        with tempfile.TemporaryDirectory() as d:
            t = default_tools(d)
            Path(d, "a.py").write_text("x = 1\n")
            t.run("read_file", {"path": "a.py"})
            t.run("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"})
            t.run("run_command", {"command": "exit 3"})
            f = t.session.facts()
        self.assertIn("files read: a.py", f)
        self.assertIn("files changed: a.py", f)
        self.assertIn("`exit 3` -> exit code 3", f)

class ArgAliases(Workspace):  # B26
    def test_line_start_line_end_accepted(self):
        out = self.tools.run("read_file", {"path": "big.py", "line_start": 5, "line_end": 6})
        self.assertEqual(out, "[big.py lines 5-6 of 800]\nline_5 = 5\nline_6 = 6\n")

    def test_real_name_wins(self):
        out = self.tools.run("read_file", {"path": "big.py", "start_line": 1, "end_line": 1, "line_start": 9})
        self.assertIn("unknown=['line_start']", out)

class DailyQuota(unittest.TestCase):  # B29: a used-up daily limit is not a temporary rate limit
    def test_per_day_429_is_quota_not_rate_limit(self):
        from runtime.model.base import ModelQuotaExceeded
        def handler(req):
            return httpx.Response(429, json={"error": {"message": "Rate limit reached ... on tokens per day (TPD): Limit 200000"}})
        with self.assertRaises(ModelQuotaExceeded):
            OpenAICompatClient("m", "http://x", 5, transport=httpx.MockTransport(handler)).complete([Message("user", "hi")])

    def test_quota_is_not_retried(self):
        from runtime.model.base import ModelQuotaExceeded
        from runtime.model.retry import RetryingClient
        waits = []
        c = RetryingClient(FakeClient([ModelQuotaExceeded("daily"), call("finish", {"answer": "x"})]), sleep=waits.append)
        with self.assertRaises(ModelQuotaExceeded):
            c.complete([])
        self.assertEqual(waits, [])

    def test_per_minute_429_is_still_rate_limit(self):
        from runtime.model.base import ModelRateLimited
        def handler(req):
            return httpx.Response(429, json={"error": {"message": "Rate limit reached ... tokens per minute (TPM): Limit 8000"}})
        with self.assertRaises(ModelRateLimited):
            OpenAICompatClient("m", "http://x", 5, transport=httpx.MockTransport(handler)).complete([Message("user", "hi")])
