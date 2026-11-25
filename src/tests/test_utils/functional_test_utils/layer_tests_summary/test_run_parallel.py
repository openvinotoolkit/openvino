# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from subprocess import CompletedProcess, list2cmdline
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
        self.discovery = list_tests_patch.start()
        self.addCleanup(list_tests_patch.stop)

    def list_tests(self, command, *, stdout, **_kwargs):
        stdout.write(self.test_list)
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
            for pattern in batch.split(":")
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

    def test_discovery_ignores_standalone_parameter_annotations(self):
        self.test_list = "Brgemm/Params.\n  simpleGemmTest/Prec=f32\n  # GetParam() = f32\n"
        runner = self.make_runner("*")
        discovered = runner._TestParallelRunner__get_test_list_by_runtime()
        self.assertEqual(list(discovered), ["Brgemm/Params.simpleGemmTest/Prec=f32"])

    def test_interrupted_batch_reports_exit_code_and_missing_test(self):
        self.test_list = "Ordinary.\n  interrupted\n  not_started\n"
        runner = self.make_runner("*")
        temp_dir = Path(self.work_dir, "temp")
        temp_dir.mkdir()
        (temp_dir / "log_10.log").write_text(
            "[==========] Running 2 tests from 1 test suite.\n"
            "[ RUN      ] Ordinary.interrupted\n",
            encoding="utf-8",
        )
        with patch.object(run_parallel.TaskManager, "exit_codes", {"log_10.log": 3221225477}), \
                self.assertLogs(run_parallel.logger, level="ERROR") as messages:
            self.assertFalse(runner.postprocess_logs())
        self.assertIn(
            "ERROR:test_parallel_runner:Process of log_10.log exited with code 3221225477 during test: Ordinary.interrupted",
            messages.output,
        )
        self.assertEqual(
            Path(self.work_dir, "logs", "not_run_tests.log").read_text(encoding="utf-8"),
            "Ordinary.not_started\n",
        )

    def test_discovery_accepts_legacy_quoted_filter(self):
        self.test_list = "Ordinary.\n  smoke_first\n"
        runner = self.make_runner('"Ordinary.*"')
        discovered = runner._TestParallelRunner__get_test_list_by_runtime()
        self.assertEqual(list(discovered), ["Ordinary.smoke_first"])
        discovery_args = [sys.executable, "--gtest_filter=Ordinary.*", "--gtest_list_tests"]
        if run_parallel.constants.IS_WIN:
            discovery_args = list2cmdline(discovery_args)
        self.assertEqual(self.discovery.call_args.args[0], discovery_args)

    def test_negative_filter_with_spaces_reaches_discovery_and_worker(self):
        negative_suffix = "-*fmt=bfyx; *"
        test_filter = f"Unique/Params.*{negative_suffix}"
        self.test_list = "Unique/Params.\n  test/data_shape=(5); fmt=other;   # GetParam() = value\n"
        for is_win, supplied_filter in product((False, True), (test_filter, f'"{test_filter}"')):
            with self.subTest(is_win=is_win, supplied_filter=supplied_filter), \
                    patch.object(run_parallel.constants, "IS_WIN", is_win), \
                    patch.object(run_parallel.TaskManager, "exit_codes", {}), \
                    patch.object(run_parallel, "Popen") as launch:
                launch.return_value.poll.return_value = 0
                launch.return_value.wait.return_value = 0
                runner = self.make_runner(supplied_filter)
                cached, uncached = runner._TestParallelRunner__get_filters()
                runner._TestParallelRunner__execute_tests(cached + uncached)

                discovery_args = [sys.executable, f"--gtest_filter={test_filter}", "--gtest_list_tests"]
                worker_args = [sys.executable, f"--gtest_filter=Unique/Params.*:{negative_suffix}"]
                if is_win:
                    discovery_args = list2cmdline(discovery_args)
                    worker_args = list2cmdline(worker_args)
                self.assertEqual(self.discovery.call_args.args[0], discovery_args)
                self.assertFalse(self.discovery.call_args.kwargs.get("shell", False))
                launch.assert_called_once()
                self.assertEqual(launch.call_args.args[0], worker_args)
                self.assertFalse(launch.call_args.kwargs["shell"])


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