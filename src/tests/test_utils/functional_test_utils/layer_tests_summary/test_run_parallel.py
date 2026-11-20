import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import run_parallel


TEST_LIST = """Ordinary.
  smoke_first
OrdinaryExtra.
  smoke_second
Typed/0.  # TypeParam = float
  smoke_third
Param/ValueParameterized.
  fourth/0  # GetParam() = 0
smoke_WholeSuite.
    without_marker
"""


class TestSuiteBatches(unittest.TestCase):
    def setUp(self):
        work_dir = tempfile.TemporaryDirectory()
        self.addCleanup(work_dir.cleanup)
        self.work_dir = work_dir.name
        list_tests_patch = patch.object(run_parallel, "run", side_effect=self.list_tests)
        list_tests_patch.start()
        self.addCleanup(list_tests_patch.stop)

    def list_tests(self, command, **_kwargs):
        (Path(self.work_dir) / "test_list.lst").write_text(self.test_list, encoding="utf-8")
        return CompletedProcess(command, 0)

    test_list = TEST_LIST

    def make_runner(self, test_filter):
        return run_parallel.TestParallelRunner(
            sys.executable, [f"--gtest_filter={test_filter}"], 2, self.work_dir, "", "suite", False
        )

    def batch_patterns(self, runner):
        cached, uncached = runner._TestParallelRunner__get_filters()
        return {
            pattern
            for batch in cached + uncached
            for pattern in shlex.split(batch)[0].split(":")
            if pattern
        }

    def test_typed_and_parameterized_discovery(self):
        runner = self.make_runner("-*.DISABLED_*")
        self.assertEqual(self.batch_patterns(runner), {
            "Ordinary.*", "OrdinaryExtra.*", "Typed/0.*", "Param/ValueParameterized.*", "smoke_WholeSuite.*"
        })
        self.assertEqual(set(runner._TestParallelRunner__get_test_list_by_runtime()), {
            "Ordinary.smoke_first", "OrdinaryExtra.smoke_second", "Typed/0.smoke_third",
            "Param/ValueParameterized.fourth/0", "smoke_WholeSuite.without_marker"
        })

    def test_positive_filter_keeps_whole_suites_compact(self):
        runner = self.make_runner("Ordinary.*:OrdinaryExtra.smoke_second:Typed/0.*:Param/*.*:*smoke*:-*.DISABLED_*")
        self.assertEqual(self.batch_patterns(runner), {
            "Ordinary.*", "OrdinaryExtra.smoke_second", "Typed/0.*",
            "Param/ValueParameterized.fourth/0", "smoke_WholeSuite.*"
        })

    def test_discovery_preserves_spaces_in_parameterized_name(self):
        self.test_list = "Unique/Params.\n  test/data_shape=(5); fmt=bfyx;   # GetParam() = value\n"
        runner = self.make_runner("*")
        discovered = runner._TestParallelRunner__get_test_list_by_runtime()
        self.assertEqual(list(discovered), ["Unique/Params.test/data_shape=(5); fmt=bfyx; "])


class TestWorkerLaunch(unittest.TestCase):
    def test_windows_replacement_worker_launches_without_shell(self):
        commands = [
            'tests.exe --gtest_filter="Suite.*"',
            'tests.exe --gtest_filter="OtherSuite.*":"AnotherSuite.*"',
        ]
        with tempfile.TemporaryDirectory() as work_dir, \
                patch.object(run_parallel.constants, "IS_WIN", True), \
                patch.object(run_parallel.TaskManager, "exit_codes", {}), \
                patch.object(run_parallel, "Popen") as launch:
            launch.return_value.poll.return_value = 0
            launch.return_value.wait.return_value = 0
            manager = run_parallel.TaskManager(commands, work_dir)
            manager.init_worker()
            self.assertTrue(manager.update_worker())
            self.assertEqual([call.args[0] for call in launch.call_args_list], commands)
            self.assertEqual([call.kwargs["shell"] for call in launch.call_args_list], [False, False])


if __name__ == "__main__":
    unittest.main()