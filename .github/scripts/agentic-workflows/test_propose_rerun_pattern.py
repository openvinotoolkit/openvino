# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from propose_rerun_pattern import validate_candidate


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


if __name__ == "__main__":
    unittest.main()
