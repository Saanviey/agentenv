import tempfile, unittest
from pathlib import Path
from runtime.evals import load_tasks, run_task, summarize
from runtime.model.base import ModelUnavailable
from tests.fakes import FakeClient, call

class Evals(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name) / "tasks" / "add"
        (d / "repo").mkdir(parents=True)
        (d / "repo" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (d / "repo" / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(add(2, 3), 5)\n")
        (d / "task.toml").write_text('prompt = "fix it"\n')
        self.task = load_tasks(tasks_dir=d.parent)[0]

    def tearDown(self):
        self._tmp.cleanup()

    def test_fix_passes(self):
        r = run_task(self.task, FakeClient([call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a + b"}),
                                            call("finish", {"answer": "fixed"})]))
        self.assertEqual((r.passed, r.tampered, r.stopped), (True, False, "finish"))

    def test_editing_tests_is_caught_and_does_not_pass(self):
        r = run_task(self.task, FakeClient([call("write_file", {"path": "test_calc.py", "content": "# gone\n"})]
                                           + [call("finish", {"answer": "tests pass now"})] * 3))
        self.assertEqual((r.passed, r.tampered, r.stopped), (False, True, "unverified"))

    def test_deleting_tests_is_caught(self):
        r = run_task(self.task, FakeClient([call("run_command", {"command": "mv test_calc.py x.txt"}),
                                            call("finish", {"answer": "done"})]))  # no file change recorded -> finish accepted
        self.assertEqual((r.passed, r.tampered), (False, True))

    def test_model_crash_is_a_fail_not_an_exception(self):
        r = run_task(self.task, FakeClient([ModelUnavailable("down")]))
        self.assertEqual((r.passed, r.finished, r.stopped), (False, False, "error"))
        self.assertIn("ModelUnavailable", r.error)

    def test_original_task_repo_untouched(self):
        run_task(self.task, FakeClient([call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a + b"}),
                                        call("finish", {"answer": "ok"})]))
        self.assertIn("a - b", (self.task.repo / "calc.py").read_text())

    def test_summary(self):
        ok = run_task(self.task, FakeClient([call("edit_file", {"path": "calc.py", "old": "a - b", "new": "a + b"}),
                                             call("finish", {"answer": "ok"})]))
        bad = run_task(self.task, FakeClient([call("finish", {"answer": "nope"})]))
        s = summarize([ok, bad])
        self.assertEqual((s["passed"], s["tasks"], s["success_rate"], s["clean"]), (1, 2, 0.5, 1))

    def test_crash_keeps_progress(self):
        r = run_task(self.task, FakeClient([call("read_file", {"path": "calc.py"}), ModelUnavailable("down")]))
        self.assertEqual(r.steps, 2)
        self.assertGreater(r.input_tokens, 0)

if __name__ == "__main__":
    unittest.main()
