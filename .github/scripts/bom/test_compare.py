# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import compare


class BomTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifact = self.root / "artifact"
        self.generated = self.artifact / "linux"
        self.gold = self.root / "boms"
        self.configuration = "ubuntu_22_04_x86_64"
        self.baseline = self.gold / self.configuration
        self.generated.mkdir(parents=True)
        self.baseline.mkdir(parents=True)

    def manifest(self, directory, component="core", files=None):
        path = directory / f"{component}.json"
        path.write_text(json.dumps({
            "component": component, "files": files if files is not None else ["runtime/lib.so"],
        }), encoding="utf-8")
        return path

    def compare(self):
        return compare.compare(self.artifact, self.gold, self.configuration)

    def test_matching_and_reordered_paths(self):
        self.manifest(self.generated, files=["b", "a"])
        self.manifest(self.baseline, files=["a", "b"])
        self.assertFalse(any(result.failure for result in self.compare()))

    def test_empty_component_is_valid(self):
        self.manifest(self.generated, files=[])
        self.manifest(self.baseline, files=[])
        self.assertFalse(any(result.failure for result in self.compare()))

    def test_missing_artifact_is_failure_even_without_gold(self):
        self.artifact.rename(self.root / "moved")
        self.baseline.rmdir()
        results = self.compare()
        self.assertIn("bom_ubuntu_22_04_x86_64", results[0].failure)

    def test_empty_artifact_is_failure(self):
        self.assertIn("no component JSON", self.compare()[0].failure)

    def test_multiple_platform_directories_are_rejected(self):
        (self.artifact / "windows").mkdir()
        self.assertIn("exactly one platform", self.compare()[0].failure)

    def test_missing_configuration_fails(self):
        self.manifest(self.generated)
        self.baseline.rmdir()
        result = self.compare()
        self.assertIn("No GOLD BOM configuration", result[-1].failure)
        self.assertIn("Initialize", result[-1].failure)

    def test_missing_configuration_does_not_hide_invalid_manifest(self):
        self.manifest(self.generated).write_text("{", encoding="utf-8")
        self.baseline.rmdir()
        self.assertTrue(any(result.failure for result in self.compare()))

    def test_all_component_differences_are_reported_before_file_comparison(self):
        self.manifest(self.generated, "new")
        self.manifest(self.baseline, "removed")
        self.manifest(self.generated, "common", ["changed"])
        self.manifest(self.baseline, "common", ["original"])
        results = self.compare()
        self.assertEqual([item.name for item in results], ["components.new", "components.removed"])
        self.assertIn("Add", results[0].failure)
        self.assertIn("Remove", results[1].failure)

    def test_reports_all_file_differences_in_all_components(self):
        for name in ("core", "samples"):
            self.manifest(self.generated, name, ["added", "same"])
            self.manifest(self.baseline, name, ["removed", "same"])
        failures = [item.failure for item in self.compare() if item.failure]
        self.assertEqual(len(failures), 2)
        for message in failures:
            self.assertIn("+ added", message)
            self.assertIn("- removed", message)
            self.assertIn("Update", message)

    def test_invalid_manifest_does_not_stop_other_component_checks(self):
        self.manifest(self.generated, "broken").write_text("{", encoding="utf-8")
        self.manifest(self.baseline, "broken")
        self.manifest(self.generated, "core", ["new"])
        self.manifest(self.baseline, "core", ["old"])
        self.assertEqual(len([item for item in self.compare() if item.failure]), 2)

    def test_manifest_schema_validation(self):
        self.manifest(self.baseline)
        path = self.generated / "core.json"
        for value in (
            [], {"component": "wrong", "files": []}, {"component": "core", "files": "file"},
            {"component": "core", "files": [1]}, {"component": "core", "files": ["x", "x"]},
        ):
            with self.subTest(value=value):
                path.write_text(json.dumps(value), encoding="utf-8")
                self.assertTrue(any(item.failure for item in self.compare()))

    def test_invalid_gold_manifest_is_reported(self):
        self.manifest(self.generated)
        self.manifest(self.baseline).write_text("{", encoding="utf-8")
        self.assertIn("Cannot read BOM manifest", self.compare()[-1].failure)

    def test_invalid_utf8_manifest_is_reported(self):
        self.manifest(self.generated).write_bytes(b"\xff")
        self.manifest(self.baseline)
        self.assertIn("Cannot read BOM manifest", self.compare()[-1].failure)

    def test_unexpected_entries_are_rejected(self):
        self.manifest(self.generated)
        self.manifest(self.baseline)
        (self.generated / "nested").mkdir()
        self.assertIn("Unexpected entry", self.compare()[0].failure)

    def test_invalid_configuration_rejected(self):
        self.assertTrue(compare.compare(self.artifact, self.gold, "../escape")[0].failure)

    def test_junit_report_contains_all_failures_and_documentation(self):
        report = self.root / "reports" / "bom.xml"
        results = [compare.Result("core", failure="added\nremoved"), compare.Result("samples", failure="changed")]
        output = io.StringIO()
        with redirect_stdout(output):
            status = compare.report_results(results, self.configuration, report)
        suite = ET.parse(report).getroot()
        self.assertEqual(status, 1)
        self.assertEqual(suite.get("failures"), "2")
        self.assertEqual(len(suite.findall("testcase/failure")), 2)
        self.assertIn(compare.DOCUMENTATION_URL, output.getvalue())
        self.assertIn("added%0Aremoved", output.getvalue())

    def test_missing_baseline_produces_failure_reports_and_nonzero_exit(self):
        self.manifest(self.generated)
        self.baseline.rmdir()
        output = io.StringIO()
        with redirect_stdout(output):
            status = compare.report_results(
                self.compare(), self.configuration, self.root / "bom.xml",
            )
        self.assertEqual(status, 1)
        self.assertIn("::error::", output.getvalue())
        suite = ET.parse(self.root / "bom.xml").getroot()
        self.assertEqual(suite.get("skipped"), "0")
        self.assertEqual(suite.get("failures"), "1")
        self.assertIsNone(suite.find("testcase/skipped"))
        self.assertIn("No GOLD BOM configuration", suite.find("testcase/failure").text)
        self.assertIn("No GOLD BOM configuration", (self.root / "bom.html").read_text(encoding="utf-8"))

    def test_html_contains_all_checks_and_escapes_external_text(self):
        output = io.StringIO()
        with redirect_stdout(output):
            compare.report_results(
                [compare.Result("core<script>", failure="Added:\n<script>alert(1)</script>&"),
                 compare.Result("samples")],
                'config<&"', self.root / "bom.xml",
            )
        html = (self.root / "bom.html").read_text(encoding="utf-8")
        self.assertIn("core&lt;script&gt;", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;&amp;", html)
        self.assertNotIn("<script>", html)
        self.assertIn("config&lt;&amp;&quot;", html)
        self.assertIn("<td>FAIL</td>", html)
        self.assertIn("<td>PASS</td>", html)
        self.assertIn(compare.DOCUMENTATION_URL, html)

    def test_cli_exit_status_and_no_baseline_modifications(self):
        self.manifest(self.generated, files=["new"])
        golden = self.manifest(self.baseline, files=["old"])
        before = golden.read_bytes()
        command = [
            sys.executable, compare.__file__, "--artifact-dir", str(self.artifact),
            "--gold-dir", str(self.gold), "--configuration", self.configuration,
            "--report", str(self.root / "bom.xml"),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("::error::", result.stdout)
        self.assertEqual(golden.read_bytes(), before)
        self.manifest(self.generated, files=["old"])
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertTrue((self.root / "bom.html").is_file())
        golden.unlink()
        self.baseline.rmdir()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("No GOLD BOM configuration", result.stdout)


if __name__ == "__main__":
    unittest.main()
