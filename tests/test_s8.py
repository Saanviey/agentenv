import json, shlex, sys, tempfile, unittest
from pathlib import Path
from runtime.checkpoint import checkpointer, decode, encode
from runtime.loop import run_agent, RunState
from runtime.memory import memory_path, add_fact, load_facts, prompt_block, MAX_FACTS
from runtime.model.base import ModelUnavailable
from runtime.policy import approve_all, check
from runtime.store import Store
from runtime.tools import default_tools
from runtime.trace import Tracer, load
from runtime.types import Message
from runtime.verify import CodingVerifier
from tests.fakes import FakeClient, call

TESTS = shlex.join([sys.executable, "-m", "unittest", "-q"])
SCRIPT = [call("read_file", {"path": "calc.py"}), call("run_command", {"command": TESTS}),
          call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a + b"}), call("run_command", {"command": TESTS}),
          call("finish", {"answer": "fixed add"})]

class Kill(BaseException): ...  # like kill -9: not an Exception, so no cleanup code runs

class Reactive(FakeClient):  # a fake model that redoes a call the runtime says was interrupted, like a sensible agent would
    def complete(self, messages, tools=None):
        last = messages[-1]
        if last.role == "tool" and "may or may not have happened" in last.content:
            prev = next(m for m in reversed(messages) if m.role == "assistant")
            c = next(c for c in prev.tool_calls if c.id == last.tool_call_id or c.name == last.tool_name)
            self.calls.append(list(messages))
            return call(c.name, c.arguments)
        return super().complete(messages, tools)

class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "ws"
        self.root.mkdir()
        (self.root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (self.root / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(add(2, 3), 5)\n")
        self.store = Store(self.dir / "agent.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def start(self, run_id="r1", task="fix it"):
        self.store.create(run_id, str(self.root), task, "fake", "")
        return [Message("system", "s"), Message("user", task)]

    def resume(self, run_id, script, max_steps=10):  # what `python -m runtime --resume` does: fresh objects, state from the database
        tools = default_tools(self.root)
        st = decode(self.store.get(run_id)["state"], tools, None)
        used = sum(m.role == "assistant" for m in st.messages)  # replies already received and saved
        client = Reactive(script[used:])
        r = run_agent(client, tools, [], max_steps, approve=approve_all, verifier=CodingVerifier(TESTS),
                      checkpoint=checkpointer(self.store, run_id, tools, None), state=st)
        return r, tools

class StoreTests(Base):
    def test_create_save_get(self):
        self.start()
        self.store.save("r1", 3, "running", {"x": 1})
        r = self.store.get("r1")
        self.assertEqual((r["status"], r["step"], r["state"], r["workspace"]), ("running", 3, {"x": 1}, str(self.root)))

    def test_latest_resumable(self):
        self.start("a"); self.store.save("a", 1, "finish", {})
        self.start("b"); self.store.save("b", 1, "crashed", {})
        self.start("c"); self.store.save("c", 1, "finish", {})
        self.assertEqual(self.store.latest(), "c")
        self.assertEqual(self.store.latest(resumable_only=True), "b")

    def test_missing(self):
        self.assertIsNone(self.store.get("nope"))

class Roundtrip(Base):
    def test_encode_decode(self):
        tools = default_tools(self.root)
        tools.run("read_file", {"path": "calc.py"})
        tools.run("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"})
        st = RunState([Message("system", "s"), Message("user", "t")] + [call("read_file", {"path": "x"}).message],
                      step=4, in_tok=10, out_tok=5, reminders=1, rejections=2, seen={"k": 1}, sent=3, pending=0)
        d = json.loads(json.dumps(encode(st, tools, None)))   # through real JSON, like the database
        tools2 = default_tools(self.root)
        st2 = decode(d, tools2, None)
        self.assertEqual(st2, st)
        self.assertEqual(tools2.session.snapshots, {"calc.py": "def add(a, b):\n    return a - b\n"})
        self.assertEqual(tools2.session.read, {"calc.py"})

class KillAnywhere(Base):  # done-when, unit level: kill at EVERY save point, resume, same ending
    def run_with_kill_at(self, k):
        msgs = self.start(f"run{k}")
        real = checkpointer(self.store, f"run{k}", (tools := default_tools(self.root)), None)
        n = {"saves": 0}
        def ck(st, status):
            real(st, status)
            if status == "running":
                n["saves"] += 1
                if n["saves"] == k:
                    raise Kill()
        try:
            r = run_agent(Reactive(list(SCRIPT)), tools, msgs, 10, approve=approve_all, verifier=CodingVerifier(TESTS), checkpoint=ck)
            return r, None
        except Kill:
            return None, self.resume(f"run{k}", SCRIPT)[0]

    def test_every_kill_point(self):
        ref, _ = self.run_with_kill_at(0)                     # 0 = never killed
        self.assertEqual((ref.stopped, ref.answer), ("finish", "fixed add"))
        k = 1
        while True:
            (self.root / "calc.py").write_text("def add(a, b):\n    return a - b\n")   # fresh buggy file each time
            done, resumed = self.run_with_kill_at(k)
            if done is not None:                              # k is past the last save point: finished without a kill
                break
            with self.subTest(kill_at_save=k):
                self.assertEqual((resumed.stopped, resumed.answer), ("finish", "fixed add"))
                self.assertIn("a + b", (self.root / "calc.py").read_text())
            k += 1
        self.assertGreater(k, 8)                              # many save points were really tested

class HalfDone(Base):
    def test_write_interrupted_mid_tool_is_not_rerun_model_is_told(self):
        msgs = self.start()
        tools = default_tools(self.root)
        orig = tools.get("edit_file").fn
        def dying_edit(**kw):                                  # the edit happens, then the process dies before saving the result
            orig(**kw)
            raise Kill()
        tools.get("edit_file").fn = dying_edit
        with self.assertRaises(Kill):
            run_agent(FakeClient(SCRIPT), tools, msgs, 10, approve=approve_all, checkpoint=checkpointer(self.store, "r1", tools, None))
        st = decode(self.store.get("r1")["state"], default_tools(self.root), None)
        self.assertEqual(st.pending, 0)
        r, _ = self.resume("r1", SCRIPT)
        notes = [m.content for m in r.messages if m.role == "tool" and "may or may not have happened" in m.content]
        self.assertEqual(len(notes), 1)
        self.assertEqual(r.stopped, "finish")
        self.assertEqual((self.root / "calc.py").read_text().count("a + b"), 1)   # applied once, not twice

    def test_read_interrupted_is_simply_rerun(self):
        msgs = self.start()
        tools = default_tools(self.root)
        tools.get("read_file").fn = lambda **kw: (_ for _ in ()).throw(Kill())
        with self.assertRaises(Kill):
            run_agent(FakeClient(SCRIPT), tools, msgs, 10, approve=approve_all, checkpoint=checkpointer(self.store, "r1", tools, None))
        r, _ = self.resume("r1", SCRIPT)
        self.assertFalse(any("may or may not have happened" in m.content for m in r.messages))
        self.assertIn("return a - b", next(m.content for m in r.messages if m.role == "tool"))

class Interrupts(Base):
    def test_ctrl_c_saves_interrupted_and_keeps_changes(self):
        msgs = self.start()
        tools = default_tools(self.root)
        path = self.dir / "t.jsonl"
        tr = Tracer(path)
        client = FakeClient([SCRIPT[2], KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            run_agent(client, tools, msgs, 10, approve=approve_all, trace=tr, verifier=CodingVerifier(TESTS),
                      checkpoint=checkpointer(self.store, "r1", tools, None))
        tr.close()
        self.assertEqual(self.store.get("r1")["status"], "interrupted")
        self.assertIn("a + b", (self.root / "calc.py").read_text())
        self.assertIn("run_interrupted", [e["event"] for e in load(path)])

    def test_crash_with_checkpoint_is_resumable_not_rolled_back(self):
        msgs = self.start()
        tools = default_tools(self.root)
        with self.assertRaises(ModelUnavailable):
            run_agent(FakeClient([call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"}), ModelUnavailable("down")]),
                      tools, msgs, 10, approve=approve_all, verifier=CodingVerifier(TESTS), checkpoint=checkpointer(self.store, "r1", tools, None))
        self.assertEqual(self.store.get("r1")["status"], "crashed")
        self.assertIn("a * b", (self.root / "calc.py").read_text())   # kept for the resume

    def test_rollback_after_resume_uses_saved_snapshot(self):
        msgs = self.start()
        tools = default_tools(self.root)
        with self.assertRaises(ModelUnavailable):
            run_agent(FakeClient([call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"}), ModelUnavailable("down")]),
                      tools, msgs, 10, approve=approve_all, verifier=CodingVerifier(TESTS), checkpoint=checkpointer(self.store, "r1", tools, None))
        r, _ = self.resume("r1", [None, call("run_command", {"command": TESTS})], max_steps=3)  # runs out of steps, tests fail
        self.assertEqual(r.stopped, "max_steps")
        self.assertIn("a - b", (self.root / "calc.py").read_text())   # rolled back to the ORIGINAL, known only from the checkpoint

class Continue(Base):
    def test_follow_up_sees_earlier_conversation(self):
        msgs = self.start()
        tools = default_tools(self.root)
        run_agent(FakeClient([call("read_file", {"path": "calc.py"}), call("finish", {"answer": "add subtracts"})]), tools, msgs,
                  checkpoint=checkpointer(self.store, "r1", tools, None))
        tools2 = default_tools(self.root)
        st = decode(self.store.get("r1")["state"], tools2, None)
        st.messages.append(Message("user", "what did you read?"))
        st.step = 0
        client = FakeClient([call("finish", {"answer": "calc.py"})])
        r = run_agent(client, tools2, [], state=st, checkpoint=checkpointer(self.store, "r1", tools2, None))
        self.assertEqual(r.answer, "calc.py")
        self.assertIn("return a - b", "\n".join(m.content for m in client.calls[0]))   # the earlier file read is in context

class Memory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = memory_path(self.dir / "proj", self.dir / "state")

    def tearDown(self):
        self._tmp.cleanup()

    def test_path_is_stable_and_outside_workspace(self):
        self.assertEqual(self.path, memory_path(self.dir / "proj", self.dir / "state"))
        self.assertNotEqual(self.path, memory_path(self.dir / "other" / "proj", self.dir / "state"))
        self.assertFalse(self.path.is_relative_to(self.dir / "proj"))

    def test_add_load_dedupe(self):
        add_fact(self.path, "tests run with  python -m unittest")
        self.assertEqual(add_fact(self.path, "Tests run with python -m unittest"), "already in project memory")
        self.assertEqual(len(load_facts(self.path)), 1)
        self.assertIn("Project memory", prompt_block(self.path))

    def test_limits(self):
        with self.assertRaises(ValueError):
            add_fact(self.path, "x" * 400)
        for i in range(MAX_FACTS):
            add_fact(self.path, f"fact {i}")
        with self.assertRaises(ValueError):
            add_fact(self.path, "one too many")

    def test_no_memory_no_block(self):
        self.assertEqual(prompt_block(self.path), "")
        self.assertEqual(prompt_block(None), "")

    def test_remember_tool_and_policy(self):
        tools = default_tools(self.dir, memory_file=self.path)
        self.assertIn("saved to project memory", tools.run("remember", {"fact": "main code is in shop/"}))
        self.assertEqual(load_facts(self.path)[0].split(" (")[0], "main code is in shop/")
        self.assertEqual(check(tools.get("remember"), {"fact": "x"}).action, "ask")
        self.assertNotIn("remember", default_tools(self.dir).names())   # no memory file -> no tool (e.g. evals)

if __name__ == "__main__":
    unittest.main()
