import json, os, tempfile, unittest
from pathlib import Path
from runtime.loop import run_agent
from runtime.model.base import ModelBadOutput, ModelUnavailable
from runtime.policy import approve_all
from runtime.tools import default_tools
from runtime.trace import Tracer, load, render, replay
from runtime.types import Message
from tests.fakes import FakeClient, call, text

class Trace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "a.py").write_text("x = 1\n")
        self.path = self.dir / "run.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def run_traced(self, script, max_steps=10):
        tr = Tracer(self.path)
        tr.event("run_start", task="t", model="fake", provider="fake", workspace=str(self.dir),
                 approval="auto", max_steps=max_steps, tools=[])
        try:
            run_agent(FakeClient(script), default_tools(self.dir), [Message("system", "s"), Message("user", "t")],
                      max_steps, approve=approve_all, trace=tr)
        finally:
            tr.close()
        return load(self.path)

    def test_every_line_is_json_with_time_and_event(self):
        self.run_traced([call("finish", {"answer": "ok"})])
        for line in self.path.read_text().splitlines():
            rec = json.loads(line)
            self.assertIn("event", rec)
            self.assertIn("t", rec)

    def test_event_sequence(self):
        ev = self.run_traced([call("read_file", {"path": "a.py"}), text("hm"), call("finish", {"answer": "ok"})])
        self.assertEqual([e["event"] for e in ev],
                         ["run_start", "model_request", "model_response", "tool",
                          "model_request", "model_response", "reminder",
                          "model_request", "model_response", "run_end"])

    def test_only_new_messages_per_request(self):
        ev = self.run_traced([call("read_file", {"path": "a.py"}), call("finish", {"answer": "ok"})])
        reqs = [e for e in ev if e["event"] == "model_request"]
        self.assertEqual([m["role"] for m in reqs[0]["new_messages"]], ["system", "user"])
        self.assertEqual([m["role"] for m in reqs[1]["new_messages"]], ["assistant", "tool"])

    def test_tool_event_has_policy_and_result(self):
        ev = self.run_traced([call("run_command", {"command": "rm -rf ../"}), call("finish", {"answer": "ok"})])
        t = next(e for e in ev if e["event"] == "tool")
        self.assertEqual((t["decision"], t["approved"]), ("deny", False))
        self.assertIn("blocked by policy", t["result"])

    def test_crash_is_recorded(self):
        with self.assertRaises(ModelUnavailable):
            self.run_traced([ModelUnavailable("down")])
        ev = load(self.path)
        self.assertEqual(ev[-1]["event"], "run_error")
        self.assertEqual(ev[-2]["type"], "ModelUnavailable")

    def test_bad_output_records_what_model_wrote(self):
        ev = self.run_traced([ModelBadOutput("400", '{"name": oops}'), call("finish", {"answer": "ok"})])
        err = next(e for e in ev if e["event"] == "model_error")
        self.assertEqual(err["generation"], '{"name": oops}')

    def test_no_secrets_in_trace(self):
        os.environ["TEST_API_KEY"] = "sk-SECRET123"
        try:
            self.run_traced([call("run_command", {"command": "env"}), call("finish", {"answer": "ok"})])
        finally:
            del os.environ["TEST_API_KEY"]
        self.assertNotIn("sk-SECRET123", self.path.read_text())

    def test_render_and_replay(self):
        self.run_traced([call("read_file", {"path": "a.py"}), call("finish", {"answer": "done"})])
        view = render(load(self.path))
        self.assertIn("read_file", view)
        self.assertIn("answer: done", view)
        self.assertIn("SAME", replay(self.path))

if __name__ == "__main__":
    unittest.main()
