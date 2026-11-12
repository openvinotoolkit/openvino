# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coverage = load("check_test_coverage")
summarizer = load("summarize_tests")


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.contract = {"selections": [{"pattern": "*", "count": 2},
                                        {"pattern": "Arch/Graph.*", "count": 1}]}
        self.listing = "Running main()\nArch/Graph. # TypeParam = int\n  Decode/0 # GetParam() = x\nAPI.\n  Reset\n"

    def summary(self, skip=False):
        xml = '<testsuites><testsuite><testcase classname="Arch/Graph" name="Decode/0"/>'
        xml += '<testcase classname="API" name="Reset">' + ('<skipped message="no model"/>' if skip else '')
        xml += '</testcase></testsuite></testsuites>'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tests.xml"
            path.write_text(xml)
            return {"process_exit": 0, "parse_errors": [], "reports": [summarizer.summarize(path)]}

    def test_real_xml_ids_match_typed_listing(self):
        self.assertEqual(coverage.check(self.contract, self.listing, self.summary())["violations"], [])

    def test_wrong_filter_and_missing_registration(self):
        self.assertTrue(coverage.check(self.contract, "API.\n  Reset\n")["violations"])
        self.assertTrue(coverage.check(self.contract, "Running main()\n")["violations"])

    def test_stale_executable_missing_new_test(self):
        summary = self.summary()
        summary["reports"][0]["selected_ids"].pop()
        summary["reports"][0]["counts"]["selected"] -= 1
        self.assertIn("listed and executed IDs differ", coverage.check(self.contract, self.listing, summary)["violations"])

    def test_skip_requires_explicit_allowance(self):
        self.assertTrue(coverage.check(self.contract, self.listing, self.summary(True))["violations"])
        self.contract["allowed_skips"] = ["API.Reset"]
        self.assertEqual(coverage.check(self.contract, self.listing, self.summary(True))["violations"], [])

    def test_process_failure_and_duplicate_reports(self):
        summary = self.summary()
        summary["process_exit"] = 1
        self.assertTrue(coverage.check(self.contract, self.listing, summary)["violations"])
        summary["process_exit"] = 0
        summary["reports"] *= 2
        self.assertTrue(coverage.check(self.contract, self.listing, summary)["violations"])

    def test_required_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            self.contract["required_files"] = [str(path)]
            self.assertTrue(coverage.check(self.contract, self.listing)["violations"])
            path.write_bytes(b"GGUF")
            self.assertEqual(coverage.check(self.contract, self.listing)["violations"], [])

    def test_invalid_count(self):
        for count in (0, True, "2"):
            with self.assertRaises(ValueError):
                coverage.check({"selections": [{"pattern": "*", "count": count}]}, self.listing)

    def test_cli_preserves_fixture_and_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fixture, contract, listing = path / "model.gguf", path / "contract.json", path / "listing.txt"
            fixture.write_bytes(b"GGUF")
            self.contract["required_files"] = [str(fixture)]
            contract.write_text(json.dumps(self.contract))
            listing.write_text(self.listing)
            original = {target: target.read_bytes() for target in (fixture, contract, listing)}
            for output in (fixture, contract, listing):
                result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                         "--listing", str(listing), "--output", str(output)], capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["output_refused"], True)
                self.assertEqual({target: target.read_bytes() for target in original}, original)

    def test_cli_malformed_contract_retains_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            contract, listing, output = path / "contract.json", path / "listing.txt", path / "result.json"
            contract.write_text('{"selections": []}')
            listing.write_text(self.listing)
            result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                     "--listing", str(listing), "--output", str(output)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            report = json.loads(output.read_text())
            self.assertTrue(report["input_error"])
            self.assertEqual(json.loads(result.stdout)["details"], str(output.resolve()))

    def test_cli_relative_required_file_and_later_fixture_alias_refuses_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model, contract, listing = path / "model.gguf", path / "contract.json", path / "listing.txt"
            original = b"MODEL BYTES MUST SURVIVE"
            model.write_bytes(original)
            contract.write_text(json.dumps({"selections": self.contract["selections"],
                                            "required_files": ["relative.gguf", str(model)]}))
            listing.write_text(self.listing)
            result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                     "--listing", str(listing), "--output", str(model)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            diagnostic = json.loads(result.stdout)
            self.assertIsNone(diagnostic["details"])
            self.assertTrue(diagnostic["output_refused"])
            self.assertIn("required fixture", diagnostic["reason"])
            self.assertEqual(model.read_bytes(), original)

    def test_cli_malformed_required_files_refuses_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            output, contract, listing = path / "result.json", path / "contract.json", path / "listing.txt"
            original = b"KEEP EXISTING OUTPUT"
            output.write_bytes(original)
            listing.write_text(self.listing)
            malformed = ("not a list", [None], [""], [str(output), None])
            for required_files in malformed:
                with self.subTest(required_files=required_files):
                    contract.write_text(json.dumps({"selections": self.contract["selections"],
                                                    "required_files": required_files}))
                    result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                             "--listing", str(listing), "--output", str(output)], capture_output=True)
                    self.assertEqual(result.returncode, 2)
                    diagnostic = json.loads(result.stdout)
                    self.assertIsNone(diagnostic["details"])
                    self.assertTrue(diagnostic["output_refused"])
                    self.assertTrue(diagnostic["reason"])
                    self.assertEqual(output.read_bytes(), original)

    def test_cli_unreadable_contract_discovery_refuses_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            contract, listing, output = path / "contract.json", path / "listing.txt", path / "result.json"
            listing.write_text(self.listing)
            original = b"KEEP EXISTING OUTPUT"
            output.write_bytes(original)
            for contents in ("{broken", "[]"):
                with self.subTest(contents=contents):
                    contract.write_text(contents)
                    result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                             "--listing", str(listing), "--output", str(output)], capture_output=True)
                    self.assertEqual(result.returncode, 2)
                    diagnostic = json.loads(result.stdout)
                    self.assertIsNone(diagnostic["details"])
                    self.assertTrue(diagnostic["output_refused"])
                    self.assertTrue(diagnostic["reason"])
                    self.assertEqual(output.read_bytes(), original)

    def test_cli_fixture_symlink_and_hardlink_aliases_refuse_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model, contract, listing = path / "model.gguf", path / "contract.json", path / "listing.txt"
            model.write_bytes(b"KEEP MODEL BYTES")
            contract.write_text(json.dumps({"selections": self.contract["selections"],
                                            "required_files": [str(model)]}))
            listing.write_text(self.listing)
            for kind in ("symlink", "hardlink"):
                with self.subTest(kind=kind):
                    output = path / (kind + "-output")
                    if kind == "symlink":
                        output.symlink_to(model)
                    else:
                        os.link(model, output)
                    result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                             "--listing", str(listing), "--output", str(output)], capture_output=True)
                    self.assertEqual(result.returncode, 2)
                    diagnostic = json.loads(result.stdout)
                    self.assertIsNone(diagnostic["details"])
                    self.assertTrue(diagnostic["output_refused"])
                    self.assertEqual(model.read_bytes(), b"KEEP MODEL BYTES")
                    self.assertEqual(output.read_bytes(), b"KEEP MODEL BYTES")
                    output.unlink()
            first_link, second_link = path / "loop-a", path / "loop-b"
            first_link.symlink_to(second_link)
            second_link.symlink_to(first_link)
            output = path / "loop-output.json"
            output.write_bytes(b"KEEP OUTPUT BYTES")
            contract.write_text(json.dumps({"selections": self.contract["selections"],
                                            "required_files": [str(first_link)]}))
            result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                     "--listing", str(listing), "--output", str(output)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            diagnostic = json.loads(result.stdout)
            self.assertIsNone(diagnostic["details"])
            self.assertTrue(diagnostic["output_refused"])
            self.assertIn("cannot safely resolve", diagnostic["reason"])
            self.assertEqual(output.read_bytes(), b"KEEP OUTPUT BYTES")

    def test_hardlink_output_preserves_coverage_inputs_and_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            contract, listing, output = path / "contract.json", path / "listing.txt", path / "result.json"
            original = json.dumps(self.contract)
            contract.write_text(original)
            listing.write_text(self.listing)
            os.link(contract, output)
            result = subprocess.run([sys.executable, coverage.__file__, "--contract", str(contract),
                                     "--listing", str(listing), "--output", str(output)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(contract.read_text(), original)
            xml, alias = path / "tests.xml", path / "xml-alias.json"
            xml.write_text('<testsuites/>')
            os.link(xml, alias)
            result = subprocess.run([sys.executable, summarizer.__file__, str(xml),
                                     "--output", str(alias)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(xml.read_text(), '<testsuites/>')


if __name__ == "__main__":
    unittest.main()
