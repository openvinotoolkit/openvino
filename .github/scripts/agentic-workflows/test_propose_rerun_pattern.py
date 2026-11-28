# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from propose_rerun_pattern import load_open_proposals, validate_candidate


class ValidateCandidateTest(unittest.TestCase):
    def test_single_line_evidence_is_accepted(self):
        self.assertEqual(
            validate_candidate(
                "Each download failed!", "Network", "abcdef12",
                "CMake Error: Each download failed!",
            ),
            [],
        )

    def test_multiline_evidence_is_rejected(self):
        for separator in ("\n", "\r", "\r\n"):
            with self.subTest(separator=repr(separator)):
                problems = validate_candidate(
                    "Each download failed!", "Network", "abcdef12",
                    f"Each download{separator}failed!",
                )
                self.assertIn("sample_log_line must be a single log line (no CR or LF)", problems)

    def test_multiline_candidate_is_rejected(self):
        for separator in ("\n", "\r", "\r\n"):
            with self.subTest(separator=repr(separator)):
                problems = validate_candidate(
                    f"Each download{separator}failed!", "Network", "abcdef12",
                    "Each download failed!",
                )
                self.assertTrue(any("search string must be" in problem for problem in problems))


class LoadOpenProposalsTest(unittest.TestCase):
    def setUp(self):
        self.repo = Mock()
        self.repo.get_pulls.return_value = [
            SimpleNamespace(head=SimpleNamespace(
                ref="ci-doctor/rerun-pattern/abcdef12", sha="headsha",
            ), number=1),
        ]
        self.path = ".github/scripts/workflow_rerun/errors_to_look_for.json"

    def test_read_error_propagates(self):
        self.repo.get_contents.side_effect = OSError("Unable to read proposal file")
        with self.assertRaisesRegex(OSError, "Unable to read proposal file"):
            load_open_proposals(self.repo, "master", self.path, set())

    def test_invalid_json_propagates(self):
        self.repo.get_contents.return_value = SimpleNamespace(
            decoded_content=b"invalid JSON", sha="blobsha",
        )
        with self.assertRaises(json.JSONDecodeError):
            load_open_proposals(self.repo, "master", self.path, set())

    def test_only_new_proposal_patterns_are_collected(self):
        self.repo.get_contents.return_value = SimpleNamespace(
            decoded_content=json.dumps([
                {"error_text": "existing pattern"},
                {"error_text": "new proposed pattern"},
            ]).encode("utf-8"),
            sha="blobsha",
        )
        self.assertEqual(
            load_open_proposals(self.repo, "master", self.path, {"existing pattern"}),
            [("open PR #1", "new proposed pattern")],
        )


if __name__ == "__main__":
    unittest.main()
