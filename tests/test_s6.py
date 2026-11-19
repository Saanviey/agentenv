import shlex, sys, tempfile, unittest
from pathlib import Path
from runtime.loop import run_agent
from runtime.model.base import ModelUnavailable
from runtime.policy import approve_all
from runtime.tools import default_tools
from runtime.types import Message
from runtime.verify import CodingVerifier, Verifier
from tests.fakes import FakeClient, call, text

TESTS = shlex.join([sys.executable, "-m", "unittest", "-q"])
FIX = call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a + b"})
RUN_TESTS = call("run_command", {"command": TESTS})

def done(answer="fixed"):
    return call("finish", {"answer": answer})

class S6(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (self.root / "notes.md").write_text("secret plans\n")
        (self.root / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(add(2, 3), 5)\n")
        self.tools = default_tools(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def run_it(self, script, verifier=None, max_steps=10, rollback=True):
        return run_agent(FakeClient(script), self.tools, [Message("system", "s"), Message("user", "t")], max_steps,
                         approve=approve_all, verifier=verifier or CodingVerifier(), rollback=rollback)

    def results(self, r):
        return [m.content for m in r.messages if m.role in ("tool", "user")][1:]

    def calc(self):
        return (self.root / "calc.py").read_text()

class Evidence(S6):
    def test_finish_after_change_without_check_is_rejected(self):
        r = self.run_it([FIX, done(), RUN_TESTS, done()])
        self.assertEqual((r.stopped, r.steps), ("finish", 4))
        self.assertTrue(any("haven't run a successful check" in m for m in self.results(r)))

    def test_failing_check_is_not_evidence(self):
        r = self.run_it([call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"}), RUN_TESTS, done(),
                         call("edit_file", {"path": "calc.py", "old": "a * b", "new": "a + b"}), RUN_TESTS, done()])
        self.assertEqual((r.stopped, r.steps), ("finish", 6))

    def test_test_command_decides(self):
        r = self.run_it([call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"}), done(),
                         call("edit_file", {"path": "calc.py", "old": "a * b", "new": "a + b"}), done()],
                        verifier=CodingVerifier(TESTS))
        self.assertEqual((r.stopped, r.steps), ("finish", 4))
        self.assertTrue(any("tests still fail" in m for m in self.results(r)))

    def test_no_changes_no_check_needed(self):
        r = self.run_it([call("read_file", {"path": "calc.py"}), done("add subtracts in calc.py")])
        self.assertEqual((r.stopped, r.steps), ("finish", 2))

    def test_unread_file_in_answer_is_rejected(self):
        r = self.run_it([done("notes.md describes the plan"), call("read_file", {"path": "notes.md"}), done("notes.md has plans")])
        self.assertEqual((r.stopped, r.steps), ("finish", 3))
        self.assertTrue(any("never read" in m for m in self.results(r)))

    def test_file_seen_in_test_output_counts(self):
        r = self.run_it([RUN_TESTS, done("test_calc.py fails")])
        self.assertEqual(r.stopped, "finish")

    def test_nonexistent_file_names_are_ignored(self):
        r = self.run_it([done("you could add utils.py")])
        self.assertEqual(r.stopped, "finish")

    def test_too_many_rejections_marks_unverified(self):
        r = self.run_it([FIX, done(), done(), done()], rollback=False)
        self.assertEqual(r.stopped, "unverified")

    def test_salvaged_finish_is_verified_too(self):
        r = self.run_it([FIX, text('{"answer": "fixed"}'), RUN_TESTS, text('{"answer": "fixed"}')])
        self.assertEqual((r.stopped, r.steps), ("finish", 4))

    def test_base_verifier_accepts_anything(self):
        r = self.run_it([FIX, done()], verifier=Verifier())
        self.assertEqual((r.stopped, r.steps), ("finish", 2))

class AutoRevert(S6):
    def test_syntax_error_edit_is_undone(self):
        out = self.tools.run("edit_file", {"path": "calc.py", "old": "return a - b", "new": "return a +"})
        self.assertIn("invalid Python", out)
        self.assertIn("a - b", self.calc())

    def test_new_file_with_syntax_error_is_removed(self):
        out = self.tools.run("write_file", {"path": "bad.py", "content": "def f(:\n"})
        self.assertIn("undone", out)
        self.assertFalse((self.root / "bad.py").exists())

    def test_non_python_files_not_syntax_checked(self):
        self.assertIn("wrote", self.tools.run("write_file", {"path": "x.txt", "content": "def f(:"}))

    def test_revert_file(self):
        self.tools.run("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"})
        self.tools.run("edit_file", {"path": "calc.py", "old": "a * b", "new": "a / b"})
        self.assertIn("restored", self.tools.run("revert_file", {"path": "calc.py"}))
        self.assertIn("a - b", self.calc())

    def test_revert_unchanged_file(self):
        self.assertIn("nothing to revert", self.tools.run("revert_file", {"path": "calc.py"}))

class Rollback(S6):
    BREAK = call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"})

    def test_failed_run_with_failing_tests_is_rolled_back(self):
        r = self.run_it([self.BREAK, call("write_file", {"path": "new.py", "content": "x = 1\n"})] + [RUN_TESTS] * 3,
                        verifier=CodingVerifier(TESTS), max_steps=5)
        self.assertEqual(r.stopped, "max_steps")
        self.assertIn("a - b", self.calc())
        self.assertFalse((self.root / "new.py").exists())

    def test_failed_run_with_passing_tests_keeps_changes(self):
        r = self.run_it([FIX, RUN_TESTS, RUN_TESTS], verifier=CodingVerifier(TESTS), max_steps=3)
        self.assertEqual(r.stopped, "max_steps")
        self.assertIn("a + b", self.calc())

    def test_crash_with_failing_tests_is_rolled_back(self):
        with self.assertRaises(ModelUnavailable):
            self.run_it([self.BREAK, ModelUnavailable("down")], verifier=CodingVerifier(TESTS))
        self.assertIn("a - b", self.calc())

    def test_no_test_command_keeps_changes(self):
        self.run_it([self.BREAK, RUN_TESTS], max_steps=2)
        self.assertIn("a * b", self.calc())

    def test_rollback_off(self):
        self.run_it([self.BREAK, RUN_TESTS], verifier=CodingVerifier(TESTS), max_steps=2, rollback=False)
        self.assertIn("a * b", self.calc())

if __name__ == "__main__":
    unittest.main()

class StaleBytecode(S6):
    def test_same_size_edit_in_same_second_is_seen_by_tests(self):  # B19
        self.tools.run("edit_file", {"path": "calc.py", "old": "a - b", "new": "a * b"})
        self.assertIn("exit code: 1", self.tools.run("run_command", {"command": TESTS}))
        self.tools.run("edit_file", {"path": "calc.py", "old": "a * b", "new": "a + b"})
        self.assertIn("exit code: 0", self.tools.run("run_command", {"command": TESTS}))
