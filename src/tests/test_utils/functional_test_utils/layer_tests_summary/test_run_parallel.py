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
        (Path(self.work_dir) / "test_list.lst").write_text(TEST_LIST, encoding="utf-8")
        return CompletedProcess(command, 0)

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


if __name__ == "__main__":
    unittest.main()