# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from check_overall_status import main, overall_status_failed, resolve_target


class OverallStatusTest(unittest.TestCase):
    @patch("check_overall_status.subprocess.run")
    def test_only_failed_overall_status_allows_investigation(self, run):
        for name, conclusion, expected in (
            ("ci/gha_overall_status", "failure", True),
            ("ci/gha_overall_status_windows", "failure", True),
            ("ci/gha_overall_status", "success", False),
            ("ci/gha_overall_status", "cancelled", False),
            ("ci/gha_overall_status", "skipped", False),
            ("ci/gha_overall_status", "timed_out", False),
            ("ci/gha_overall_status", None, False),
            ("Unrelated job", "failure", False),
            ("ci/gha_overall_status_extra / Test", "failure", False),
        ):
            with self.subTest(name=name, conclusion=conclusion):
                run.return_value.stdout = json.dumps([
                    {"jobs": [{"name": "Optional job", "conclusion": "failure"}]},
                    {"jobs": [{"name": name, "conclusion": conclusion}]},
                ])
                self.assertEqual(overall_status_failed("owner/repo", "123", "2"), expected)
        self.assertIn(
            "repos/owner/repo/actions/runs/123/attempts/2/jobs?filter=latest&per_page=100",
            run.call_args.args[0],
        )

    @patch("check_overall_status.subprocess.run")
    def test_missing_job_skips_and_api_errors_propagate(self, run):
        run.return_value.stdout = '[{"jobs": []}]'
        self.assertFalse(overall_status_failed("owner/repo", "123", ""))
        self.assertIn("repos/owner/repo/actions/runs/123/jobs?filter=latest&per_page=100", run.call_args.args[0])
        run.side_effect = subprocess.CalledProcessError(1, ["gh", "api"])
        with self.assertRaises(subprocess.CalledProcessError):
            overall_status_failed("owner/repo", "123", "")

    def test_manual_target_resolution(self):
        with patch.dict(os.environ, {"GITHUB_SERVER_URL": "https://github.com"}):
            for link, expected in (
                ("https://github.com/other/repo/actions/runs/456", ("other/repo", "456", "")),
                ("https://github.com/other/repo/actions/runs/456/attempts/2", ("other/repo", "456", "2")),
            ):
                with self.subTest(link=link):
                    self.assertEqual(resolve_target("owner/repo", "", link, ""), expected)
            for link in (
                "https://example.com/other/repo/actions/runs/456",
                "https://github.com/other/repo/issues/456",
            ):
                with self.subTest(link=link), self.assertRaises(SystemExit):
                    resolve_target("owner/repo", "", link, "")
        self.assertEqual(resolve_target("owner/repo", "123", "ignored", "2"), ("owner/repo", "123", "2"))
        with self.assertRaises(SystemExit):
            resolve_target("owner/repo", "../jobs", "", "")

    @patch("check_overall_status.overall_status_failed")
    def test_gate_output_and_no_target(self, failed):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict(os.environ, {
                "REPO": "owner/repo", "RUN_ID": "123", "LINK": "", "RUN_ATTEMPT": "2",
                "GITHUB_OUTPUT": str(output),
            }):
                for result in (True, False):
                    failed.return_value = result
                    output.write_text("", encoding="utf-8")
                    main()
                    self.assertEqual(output.read_text(encoding="utf-8"), f"should_investigate={str(result).lower()}\n")
                failed.reset_mock()
                output.write_text("", encoding="utf-8")
                with patch.dict(os.environ, {"RUN_ID": ""}):
                    main()
                failed.assert_not_called()
                self.assertEqual(output.read_text(encoding="utf-8"), "should_investigate=false\n")


if __name__ == "__main__":
    unittest.main()
