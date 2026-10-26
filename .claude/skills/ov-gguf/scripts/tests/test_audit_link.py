# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "audit_link.py"


class AuditLinkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cwd = self.root / "build"
        self.cwd.mkdir()
        self.command = self.root / "command.txt"
        self.contract = self.root / "contract.json"

    def tearDown(self):
        self.temp.cleanup()

    def write_contract(self, required=None, forbidden=None):
        data = {
            "required_objects": ["lib/required.o"] if required is None else required,
            "forbidden_objects": [] if forbidden is None else forbidden,
        }
        self.contract.write_text(json.dumps(data), encoding="utf-8")

    def run_audit(self, command, output=None):
        self.command.write_text(command, encoding="utf-8")
        output = self.root / "report.json" if output is None else Path(output)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--command-file", str(self.command),
             "--directory", str(self.cwd), "--contract", str(self.contract),
             "--output", str(output)],
            cwd=self.root, text=True, capture_output=True, check=False,
        )
        summary = json.loads(result.stdout) if result.stdout else {}
        report = summary if summary["details"] is None else json.loads(
            Path(summary["details"]).read_text(encoding="utf-8")
        )
        return result, summary, report

    def test_missing_required_object_fails(self):
        self.write_contract()
        result, summary, report = self.run_audit("c++ lib/other.o -o app")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(summary["violations"], 1)
        self.assertEqual(report["violations"][0]["kind"], "required_object_count")
        self.assertEqual(report["violations"][0]["actual"], 0)

    def test_duplicate_object_fails(self):
        self.write_contract()
        result, summary, report = self.run_audit("c++ lib/required.o lib/required.o -o app")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(summary["violations"], 2)
        self.assertEqual(report["objects"][0]["count"], 2)
        self.assertEqual({item["kind"] for item in report["violations"]},
                         {"duplicate_object", "required_object_count"})

    def test_forbidden_object_fails(self):
        self.write_contract(forbidden=["lib/forbidden.obj"])
        result, _, report = self.run_audit("c++ lib/required.o lib/forbidden.obj -o app")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["violations"][0]["kind"], "forbidden_object")

    def test_nested_responses_use_command_directory(self):
        self.write_contract()
        response_dir = self.cwd / "args"
        response_dir.mkdir()
        (response_dir / "outer.rsp").write_text("@nested.rsp", encoding="utf-8")
        (self.cwd / "nested.rsp").write_text("lib/required.o -o app", encoding="utf-8")
        result, _, report = self.run_audit("c++ @args/outer.rsp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["response_files"], 2)

    def test_wl_response_is_expanded(self):
        self.write_contract()
        (self.cwd / "link.rsp").write_text("lib/required.o", encoding="utf-8")
        result, _, report = self.run_audit("c++ -Wl,@link.rsp -o app")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["counts"]["object_occurrences"], 1)

    def test_xlinker_object_is_counted_and_option_values_are_skipped(self):
        self.write_contract()
        command = (
            "c++ lib/required.o -Xlinker -o -Xlinker output.o "
            "-Xlinker -T -Xlinker lib/script.o"
        )
        result, _, report = self.run_audit(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["counts"]["object_occurrences"], 1)

    def test_missing_response_is_parse_error(self):
        self.write_contract()
        result, summary, _ = self.run_audit("c++ @missing.rsp")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(summary["parse_errors"], 1)
        self.assertIn("cannot read response file", summary["first_violations"][0]["message"])

    def test_response_cycle_is_parse_error(self):
        self.write_contract()
        (self.cwd / "first.rsp").write_text("@second.rsp", encoding="utf-8")
        (self.cwd / "second.rsp").write_text("@first.rsp", encoding="utf-8")
        result, summary, _ = self.run_audit("c++ @first.rsp")
        self.assertEqual(result.returncode, 2)
        self.assertIn("response-file cycle", summary["first_violations"][0]["message"])

    def test_incomplete_response_discovery_refuses_output_after_xlinker_error(self):
        self.write_contract()
        response = self.cwd / "args.rsp"
        original = b"lib/required.o\n"
        response.write_bytes(original)
        result, summary, _ = self.run_audit("c++ @args.rsp -Xlinker", output=response)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(response.read_bytes(), original)
        self.assertEqual(summary["parse_errors"], 1)
        self.assertIn("-Xlinker is missing its argument", summary["first_violations"][0]["message"])
        self.assertIsNone(summary["details"])
        self.assertEqual(summary["output_refusal_reason"], "response-file discovery was incomplete")

    def test_incomplete_nested_response_discovery_refuses_output(self):
        self.write_contract()
        (self.cwd / "outer.rsp").write_text('@nested.rsp "unterminated', encoding="utf-8")
        nested = self.cwd / "nested.rsp"
        original = b"preserve nested response\n"
        nested.write_bytes(original)
        result, summary, _ = self.run_audit("c++ @outer.rsp", output=nested)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(nested.read_bytes(), original)
        self.assertEqual(summary["parse_errors"], 1)
        self.assertIn("cannot parse response file", summary["first_violations"][0]["message"])
        self.assertIsNone(summary["details"])
        self.assertEqual(summary["output_refusal_reason"], "response-file discovery was incomplete")

    def test_shell_operators_and_substitution_are_rejected(self):
        self.write_contract()
        for command in ("c++ lib/required.o; touch marker", 'c++ "$(touch marker)" lib/required.o'):
            with self.subTest(command=command):
                result, _, report = self.run_audit(command)
                self.assertEqual(result.returncode, 2)
                self.assertTrue(report["parse_errors"])
                self.assertFalse((self.root / "marker").exists())

    def test_output_collision_preserves_command_and_refuses_output(self):
        self.write_contract()
        original = "c++ lib/required.o"
        self.command.write_text(original, encoding="utf-8")
        result, summary, report = self.run_audit(original, output=self.command)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.command.read_text(encoding="utf-8"), original)
        self.assertIsNone(summary["details"])
        self.assertTrue(any("aliases" in issue["message"] for issue in summary["first_violations"]))
        self.assertEqual(report["parse_errors"], 1)

    def test_output_collision_preserves_contract(self):
        self.write_contract()
        original = self.contract.read_text(encoding="utf-8")
        result, summary, report = self.run_audit("c++ lib/required.o", output=self.contract)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.contract.read_text(encoding="utf-8"), original)
        self.assertIsNone(summary["details"])
        self.assertTrue(any("aliases" in issue["message"] for issue in summary["first_violations"]))
        self.assertEqual(report["parse_errors"], 1)

    def test_output_collision_preserves_response(self):
        self.write_contract()
        response = self.cwd / "args.rsp"
        response.write_text("lib/required.o", encoding="utf-8")
        original = response.read_text(encoding="utf-8")
        result, summary, report = self.run_audit("c++ @args.rsp", output=response)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(response.read_text(encoding="utf-8"), original)
        self.assertIsNone(summary["details"])
        self.assertTrue(any("aliases" in issue["message"] for issue in summary["first_violations"]))
        self.assertEqual(report["parse_errors"], 1)

    def test_malformed_required_list_is_parse_error(self):
        self.write_contract(required=[])
        result, _, report = self.run_audit("c++ lib/required.o")
        self.assertEqual(result.returncode, 2)
        self.assertIn("required_objects must be a nonempty list", report["parse_errors"][0])


if __name__ == "__main__":
    unittest.main()
