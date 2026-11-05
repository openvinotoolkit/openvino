# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import summarize_tests


class SummarizeTestsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.report = self.directory / "tests.xml"
        self.output = self.directory / "summary.json"

    def run_summary(self, *reports):
        stdout = io.StringIO()
        args = ["summarize_tests.py", *map(str, reports), "--output", str(self.output), "--process-exit", "1"]
        with patch.object(sys, "argv", args), redirect_stdout(stdout):
            status = summarize_tests.main()
        return status, json.loads(stdout.getvalue()), json.loads(self.output.read_text())

    def test_namespaced_report_preserves_details(self):
        self.report.write_text("""<testsuites xmlns="urn:junit"><testsuite>
            <testcase classname="Suite" name="pass"/>
            <testcase classname="Suite" name="fail"><failure>expected &lt; actual</failure></testcase>
            <testcase classname="Suite" name="error"><error message="setup failed"/></testcase>
            <testcase classname="Suite" name="skip"><skipped message="unavailable"/></testcase>
            <testcase classname="Suite" name="disabled" status="notrun"/>
            <testcase classname="Suite" name="suppressed" result="suppressed"/>
            <error message="suite error"/>
        </testsuite></testsuites>""")
        status, compact, full = self.run_summary(self.report)
        self.assertEqual(status, 0)
        counts = {"selected": 4, "passed": 1, "failed": 1, "errors": 1, "skipped": 1, "not_run": 2}
        self.assertEqual(full, {
            "process_exit": 1,
            "counts": counts,
            "reports": [{
                "path": str(self.report.resolve()),
                "counts": counts,
                "failures": [
                    {"id": "Suite.fail", "kind": "failure", "message": "expected < actual", "signature": "expected < actual"},
                    {"id": "Suite.error", "kind": "error", "message": "setup failed", "signature": "setup failed"},
                ],
                "skips": [{"id": "Suite.skip", "message": "unavailable"}],
                "suite_errors": [{"kind": "error", "message": "suite error"}],
            }],
            "parse_errors": [],
        })
        self.assertEqual(compact, {
            "process_exit": 1, "reports": 1, "counts": counts, "suite_errors": 1,
            "failure_signatures": [
                {"count": 1, "signature": "expected < actual"},
                {"count": 1, "signature": "setup failed"},
            ],
            "omitted_signature_groups": 0, "parse_errors": 0, "details": str(self.output.resolve()),
        })

    def test_plain_gtest_and_empty_report(self):
        for xml, selected in (
            ('<testsuite><testcase name="pass" status="run" result="completed"/></testsuite>', 1),
            ("<testsuites/>", 0),
        ):
            with self.subTest(xml=xml):
                self.report.write_text(xml)
                status, compact, full = self.run_summary(self.report)
                self.assertEqual(status, 0)
                self.assertEqual(compact["counts"]["selected"], selected)
                self.assertEqual(full["counts"]["passed"], selected)

    def test_entities_are_reported_as_parse_errors(self):
        local_file = self.directory / "external.txt"
        local_file.write_text("external entity contents")
        valid = self.directory / "valid.xml"
        valid.write_text('<testsuite><testcase name="pass"/></testsuite>')
        for declaration, error in (
            ('<!ENTITY example "expanded contents">', "EntitiesForbidden"),
            (f'<!ENTITY example SYSTEM "{local_file.as_uri()}">', "EntitiesForbidden"),
            ('<!ENTITY % example "parameter contents">%example;', "EntitiesForbidden"),
        ):
            with self.subTest(declaration=declaration):
                self.report.write_text(
                    f'<!DOCTYPE testsuite [{declaration}]>'
                    '<testsuite><testcase name="example">&example;</testcase></testsuite>'
                )
                status, compact, full = self.run_summary(self.report, valid)
                self.assertEqual(status, 2)
                self.assertEqual(compact["parse_errors"], 1)
                self.assertEqual(compact["reports"], 1)
                self.assertEqual(full["counts"]["passed"], 1)
                self.assertEqual(full["parse_errors"][0]["path"], str(self.report))
                self.assertIn(error, full["parse_errors"][0]["error"])
                self.assertEqual(full["reports"][0]["path"], str(valid.resolve()))
                self.assertNotIn("external entity contents", json.dumps(full))

    def test_malformed_and_missing_reports(self):
        self.report.write_text("<testsuite>")
        missing = self.directory / "missing.xml"
        status, compact, full = self.run_summary(self.report, missing)
        self.assertEqual(status, 2)
        self.assertEqual(compact["parse_errors"], 2)
        self.assertEqual(full["reports"], [])
        self.assertEqual([error["path"] for error in full["parse_errors"]], [str(self.report), str(missing)])
        self.assertTrue(all(error["error"] for error in full["parse_errors"]))

    def test_output_cannot_overwrite_report(self):
        original = "<testsuite/>"
        self.report.write_text(original)
        self.output = self.report
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                self.run_summary(self.report)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.report.read_text(), original)


if __name__ == "__main__":
    unittest.main()
