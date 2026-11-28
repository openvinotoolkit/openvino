# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_parallel.py"
TESTS = {
    "Alpha": ("Keep", "SkipOne", "Has-Value", "Has--Value", "Has+Value", "Special-(case)"),
    "Beta": ("Keep", "SkipOne", "Other"),
    "Gamma": ("Keep", "batch=-1", "batch=1", "batch=-31", "fmt=bfyx; value", "fmt=bfyx, value"),
}


def fake_gtest():
    test_filter = os.environ.get("GTEST_FILTER", "*")
    for arg in sys.argv:
        if arg.startswith("--gtest_filter="):
            test_filter = arg.split("=", 1)[1]
        if arg.startswith("--input_folders="):
            filelist = arg.split("=", 1)[1].rstrip(",")
            if not Path(filelist).is_file():
                raise ValueError(f"Invalid input file list: {filelist}")
    positive, separator, negative = test_filter.partition("-")
    if separator and not positive:
        positive = "*"

    def matches(name, patterns):
        for pattern in patterns.split(":"):
            regex = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
            if re.fullmatch(regex, name):
                return True
        return False

    selected = {
        suite: [test for test in tests if matches(f"{suite}.{test}", positive) and not matches(f"{suite}.{test}", negative)] for suite, tests in TESTS.items()
    }
    if "--gtest_list_tests" in sys.argv:
        for suite, tests in selected.items():
            if tests:
                print(f"{suite}.")
                for test in tests:
                    print(f"  {test}")
        return

    print(f"Note: Google Test filter = {test_filter}")
    for suite, tests in selected.items():
        for test in tests:
            name = f"{suite}.{test}"
            if "--fake-gtest-drop-one" in sys.argv and name == "Alpha.Keep":
                continue
            if "--fake-gtest-drop-once" in sys.argv and name in ("Alpha.Has+Value", "Gamma.batch=-1"):
                marker = Path(os.environ["PARALLEL_FILTER_TEST_MARKER"])
                if not marker.exists():
                    marker.touch()
                    continue
            print(f"[ RUN      ] {name}")
            print(f"[       OK ] {name} (1 ms)")


class ParallelFilterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="parallel filters ")
        self.addCleanup(self.temporary.cleanup)
        self.work_dir = Path(self.temporary.name)

    def run_runner(
        self,
        test_filter=None,
        split_unit="suite",
        drop_one=False,
        repeat_failed=0,
        drop_once=False,
        env_filter=None,
        exec_file=None,
        input_folder=None,
        workers=2,
    ):
        command = [
            sys.executable,
            str(RUNNER),
            "-e",
            exec_file or sys.executable,
            "-w",
            str(self.work_dir),
            "-j",
            str(workers),
            "-s",
            split_unit,
            "-rf",
            str(repeat_failed),
            "--",
            f'"{Path(__file__).resolve()}"',
            "--fake-gtest",
        ]
        if drop_one:
            command.append("--fake-gtest-drop-one")
        if drop_once:
            command.append("--fake-gtest-drop-once")
        if input_folder is not None:
            command.append(f'--input_folders="{input_folder}"')
        if test_filter is not None:
            command.append(f"--gtest_filter={test_filter}")
        env = os.environ.copy()
        env.pop("GTEST_FILTER", None)
        if env_filter is not None:
            env["GTEST_FILTER"] = env_filter
        env["PARALLEL_FILTER_TEST_MARKER"] = str(self.work_dir / "marker")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env, cwd=self.work_dir)
        table = self.work_dir / "logs" / "hash_table.csv"
        if not table.exists():
            return result, set()
        with table.open(newline="", encoding="utf-8") as file:
            executed = {row["Test Name"] for row in csv.DictReader(file)}
        return result, executed

    def assert_selected(self, test_filter, expected, split_unit="suite"):
        result, executed = self.run_runner(test_filter, split_unit)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, set(expected), result.stdout + result.stderr)
        self.assertNotIn("Number of suites in", result.stderr)

    def test_negative_only_and_multiple_positive_and_negative_patterns(self):
        self.assert_selected(
            "-*.Skip*",
            (
                "Alpha.Keep",
                "Alpha.Has-Value",
                "Alpha.Has--Value",
                "Alpha.Has+Value",
                "Alpha.Special-(case)",
                "Beta.Keep",
                "Beta.Other",
                "Gamma.Keep",
                "Gamma.batch=-1",
                "Gamma.batch=1",
                "Gamma.batch=-31",
                "Gamma.fmt=bfyx; value",
                "Gamma.fmt=bfyx, value",
            ),
        )
        self.assert_selected(
            "Alpha.*:Beta.*-*.Skip*:*.Other",
            ("Alpha.Keep", "Alpha.Has-Value", "Alpha.Has--Value", "Alpha.Has+Value", "Alpha.Special-(case)", "Beta.Keep"),
        )

    def test_negative_filter_with_shell_characters_and_hyphens(self):
        self.assert_selected(
            "*:-*.Skip*:*.Special-(case)",
            (
                "Alpha.Keep",
                "Alpha.Has-Value",
                "Alpha.Has--Value",
                "Alpha.Has+Value",
                "Beta.Keep",
                "Beta.Other",
                "Gamma.Keep",
                "Gamma.batch=-1",
                "Gamma.batch=1",
                "Gamma.batch=-31",
                "Gamma.fmt=bfyx; value",
                "Gamma.fmt=bfyx, value",
            ),
        )
        self.assert_selected("Alpha.Keep-*--input_folders*", ("Alpha.Keep",))

    def test_exact_and_question_mark_positive_patterns(self):
        self.assert_selected("Alpha.K?ep:Beta.Keep-*.Skip*", ("Alpha.Keep", "Beta.Keep"))
        self.assert_selected("*pha.K?ep-*.Skip*", ("Alpha.Keep",))

    def test_environment_filter_and_command_line_override(self):
        test_filter = "Alpha.Has*-*.Has-Value:*.Has--Value"
        result, executed = self.run_runner(env_filter=test_filter)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, {"Alpha.Has+Value"})
        result, executed = self.run_runner("Beta.Keep", env_filter=test_filter)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, {"Beta.Keep"})

    @unittest.skipIf(os.name == "nt", "Creating executable symlinks requires extra privileges on Windows")
    def test_executable_path_with_spaces(self):
        executable = self.work_dir / "python executable"
        executable.symlink_to(sys.executable)
        result, executed = self.run_runner("Beta.Keep", exec_file=str(executable))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, {"Beta.Keep"})

    def test_quoted_filter_from_conformance_runner(self):
        self.assert_selected(
            '"Alpha.*-*.Skip*:*.Special-(case)"',
            ("Alpha.Keep", "Alpha.Has-Value", "Alpha.Has--Value", "Alpha.Has+Value"),
        )

    def test_filter_immediately_after_input_folder(self):
        with tempfile.TemporaryDirectory(prefix="parallel_models_") as input_folder:
            result, executed = self.run_runner("Beta.Keep-*.Other", input_folder=input_folder)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(executed, {"Beta.Keep"})

    def test_exclusion_is_preserved_in_test_split(self):
        self.assert_selected("Alpha.Has*-*.Has-Value:*.Has--Value", ("Alpha.Has+Value",), "test")

    def test_test_split_does_not_broaden_the_positive_filter(self):
        self.assert_selected("Alpha.Has+Value-*.Skip*", ("Alpha.Has+Value",), "test")
        self.assert_selected("Alpha.Has?Value-*.Has+Value", ("Alpha.Has-Value",), "test")

    def test_partial_filter_distinguishes_negative_batch(self):
        for split_unit in ("suite", "test"):
            with self.subTest(split_unit=split_unit):
                self.assert_selected("Gamma.batch=?1", ("Gamma.batch=-1",), split_unit)

    def test_partial_filter_preserves_shell_punctuation(self):
        self.assert_selected("Gamma.fmt=bfyx; value", ("Gamma.fmt=bfyx; value",))

    def test_device_text_in_filter_does_not_change_worker_device(self):
        result, executed = self.run_runner("Alpha.Keep:Beta.Keep-*--device*", "test", workers=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, {"Alpha.Keep", "Beta.Keep"})

    def test_retry_preserves_exclusion(self):
        result, executed = self.run_runner("Alpha.Has*-*.Has-Value:*.Has--Value", "test", repeat_failed=1, drop_once=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, {"Alpha.Has+Value"}, result.stdout + result.stderr)

    def test_retry_distinguishes_negative_batch(self):
        result, executed = self.run_runner("Gamma.batch=?1", repeat_failed=1, drop_once=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, {"Gamma.batch=-1"}, result.stdout + result.stderr)
        self.assertIn("Execute not runned 1 tests", result.stderr)

    def test_cached_suites_keep_positive_and_negative_filters(self):
        self.assert_selected(
            None,
            tuple(f"{suite}.{test}" for suite, tests in TESTS.items() for test in tests),
        )
        self.assert_selected(
            "Alpha.*:Beta.Keep-*.Skip*:*.Special-(case)",
            ("Alpha.Keep", "Alpha.Has-Value", "Alpha.Has--Value", "Alpha.Has+Value", "Beta.Keep"),
        )

    def test_reports_selected_tests_that_did_not_run(self):
        result, executed = self.run_runner("Alpha.Keep", drop_one=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(executed, set())
        self.assertIn("Alpha.Keep", (self.work_dir / "logs" / "not_run_tests.log").read_text())


if __name__ == "__main__":
    if "--fake-gtest" in sys.argv:
        fake_gtest()
    else:
        unittest.main()
