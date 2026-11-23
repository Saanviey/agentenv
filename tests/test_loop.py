import unittest
from runtime.loop import run_agent, REMINDER, CUT_OFF
from runtime.tools import default_tools
from runtime.types import Message
from runtime.model.base import ModelBadOutput
from tests.fakes import FakeClient, call, text

START = [Message("system", "s"), Message("user", "task")]

def run(script, **kw):
    client = FakeClient(script)
    return run_agent(client, default_tools("."), START, **kw), client

def tool_results(result):  # results of real tools (an accepted finish also gets a "finish accepted" result, B30)
    return [m.content for m in result.messages if m.role == "tool" and m.tool_name != "finish"]

# S2 done-when: bad JSON / unknown tool / wrong args never crash the run.
class NeverCrash(unittest.TestCase):
    def test_unknown_tool(self):
        r, _ = run([call("delete_everything"), call("finish", {"answer": "ok"})])
        self.assertEqual(r.stopped, "finish")
        self.assertIn("unknown tool 'delete_everything'", tool_results(r)[0])

    def test_missing_arg(self):
        r, _ = run([call("read_file", {}), call("finish", {"answer": "ok"})])
        self.assertIn("missing=['path']", tool_results(r)[0])

    def test_unknown_arg(self):
        r, _ = run([call("read_file", {"path": "x", "mode": "w"}), call("finish", {"answer": "ok"})])
        self.assertIn("unknown=['mode']", tool_results(r)[0])

    def test_broken_json_args(self):
        r, _ = run([call("read_file", '{"path": "con'), call("finish", {"answer": "ok"})])
        self.assertIn("not valid JSON", tool_results(r)[0])

    def test_args_not_an_object(self):
        r, _ = run([call("read_file", ["config.toml"]), call("finish", {"answer": "ok"})])
        self.assertIn("must be a JSON object", tool_results(r)[0])

    def test_path_outside_workspace(self):
        r, _ = run([call("read_file", {"path": "../../etc/passwd"}), call("finish", {"answer": "ok"})])
        self.assertIn("outside the workspace", tool_results(r)[0])

    def test_finish_without_answer(self):
        r, _ = run([call("finish", {}), call("finish", {"answer": "ok"})])
        self.assertIn("finish needs", [m.content for m in r.messages if m.role == "tool"][0])
        self.assertEqual(r.answer, "ok")

class Stopping(unittest.TestCase):
    def test_finish_ends_run(self):
        r, c = run([call("finish", {"answer": "done"})])
        self.assertEqual((r.answer, r.stopped, r.steps, len(c.calls)), ("done", "finish", 1, 1))

    def test_plain_text_gets_reminder_then_gives_up(self):
        r, _ = run([text("a"), text("b"), text("c")])
        self.assertEqual(r.stopped, "no_finish")
        self.assertEqual(sum(m.content == REMINDER for m in r.messages), 2)

    def test_reminder_counter_resets_after_a_tool_call(self):
        r, _ = run([text("a"), text("b"), call("get_time"), text("c"), call("finish", {"answer": "ok"})])
        self.assertEqual(r.stopped, "finish")

    def test_cut_off_reply_gets_specific_message(self):
        r, _ = run([text("long...", stop="length"), call("finish", {"answer": "short"})])
        self.assertIn(CUT_OFF, [m.content for m in r.messages])

    def test_max_steps(self):
        r, _ = run([call("get_time")] * 3, max_steps=3)
        self.assertEqual((r.stopped, r.steps), ("max_steps", 3))

class BadOutput(unittest.TestCase):
    def test_server_rejected_call_is_shown_to_model(self):
        r, c = run([ModelBadOutput("400", '{"name": "answer", "arguments": oops}'), call("finish", {"answer": "ok"})])
        self.assertEqual(r.stopped, "finish")
        self.assertIn("oops", c.calls[1][-1].content)

    def test_gives_up_after_repeated_bad_calls(self):
        with self.assertRaises(ModelBadOutput):
            run([ModelBadOutput("400", "x")] * 3)

class FinishAliases(unittest.TestCase):
    def test_server_rejected_alias_is_salvaged(self):
        r, _ = run([ModelBadOutput("400", '{"name": "commentary", "arguments": {"answer": "fixed it"}}')])
        self.assertEqual((r.stopped, r.answer), ("finish", "fixed it"))

    def test_plain_text_answer_json_is_salvaged(self):
        r, _ = run([text('{"answer": "All tests pass."}')])
        self.assertEqual((r.stopped, r.answer), ("finish", "All tests pass."))

    def test_structured_alias_call(self):
        r, _ = run([call("final", {"answer": "done"})])
        self.assertEqual((r.stopped, r.answer), ("finish", "done"))

    def test_other_tool_in_text_is_not_a_finish(self):
        r, _ = run([text('{"name": "read_file", "arguments": {"answer": "x"}}'), call("finish", {"answer": "ok"})])
        self.assertEqual(r.steps, 2)

    def test_nested_wrapper_is_unwrapped(self):
        gen = '{"name": "json", "arguments": {"name": "finish", "arguments": {"answer": "all good"}}}'
        r, _ = run([ModelBadOutput("400", gen)])
        self.assertEqual((r.stopped, r.answer), ("finish", "all good"))

    def test_unknown_structured_call_without_answer_is_not_finish(self):
        r, _ = run([call("final", {"text": "x"}), call("finish", {"answer": "ok"})])
        self.assertEqual(r.steps, 2)

    def test_structured_alias_with_nested_finish(self):  # B27
        r, _ = run([call("json", {"name": "finish", "arguments": {"answer": "nested"}})])
        self.assertEqual((r.stopped, r.answer), ("finish", "nested"))

    def test_broken_json_is_not_salvaged(self):
        r, _ = run([ModelBadOutput("400", '{"name": "answer", "arguments": All tests pass."}'), call("finish", {"answer": "ok"})])
        self.assertEqual(r.steps, 2)

    def test_empty_answer_is_not_salvaged(self):
        r, _ = run([text('{"answer": "  "}'), call("finish", {"answer": "ok"})])
        self.assertEqual(r.steps, 2)

class Repeats(unittest.TestCase):
    def test_repeat_call_gets_note(self):
        r, _ = run([call("list_dir", {"path": "runtime"}), call("list_dir", {"path": "runtime"}),
                    call("finish", {"answer": "ok"})])
        first, second = tool_results(r)
        self.assertNotIn("already made this exact call", first)
        self.assertIn("already made this exact call 1 time", second)

    def test_different_args_are_not_repeats(self):
        r, _ = run([call("list_dir", {"path": "runtime"}), call("list_dir", {"path": "."}),
                    call("finish", {"answer": "ok"})])
        self.assertTrue(all("already made" not in t for t in tool_results(r)))

class History(unittest.TestCase):
    def test_does_not_mutate_callers_list(self):
        start = list(START)
        run_agent(FakeClient([call("finish", {"answer": "ok"})]), default_tools("."), start)
        self.assertEqual(len(start), 2)

    def test_model_sees_tool_result_next_step(self):
        _, c = run([call("read_file", {"path": "config.toml"}), call("finish", {"answer": "ok"})])
        self.assertIn("[agent]", c.calls[1][-1].content)

if __name__ == "__main__":
    unittest.main()
