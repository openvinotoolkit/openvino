# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks using tiny subprocesses and an isolated Git repository."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


RUNNER = Path(__file__).with_name("run_matrix.py")


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.source = self.repo / "source.txt"
        self.source.write_text("original")
        self.git("add", "source.txt")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
        self.model = self.root / "model.bin"
        self.model.write_bytes(b"model")
        self.manifest = self.root / "matrix.json"
        self.output = self.root / "results"
        self.data = {"version": 1, "context": {"build": "test"}, "repositories": [str(self.repo)],
                     "inputs": [], "cases": [self.case("pass", "print('ok')")]}

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def case(self, name, code, timeout=5):
        return {"id": name, "cwd": str(self.repo), "command": [sys.executable, "-c", code],
                "timeout_seconds": timeout, "inputs": [str(self.model)],
                "metadata": {"acceptance": "exit zero"}}

    def command(self):
        self.manifest.write_text(json.dumps(self.data))
        return [sys.executable, str(getattr(self, "runner", RUNNER)), str(self.manifest), "--output", str(self.output)]

    def run_batch(self, expected=0, *extra):
        result = subprocess.run(self.command() + list(extra), capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def record(self, name="pass"):
        return json.loads((self.output / f"{name}.json").read_text())

    def test_resume_and_input_invalidation(self):
        self.run_batch()
        first = self.record()
        self.run_batch()
        self.assertEqual(self.record()["log"], first["log"])
        self.model.write_bytes(b"changed")
        self.run_batch()
        self.assertNotEqual(self.record()["log"], first["log"])
        second = self.record()
        self.source.write_text("modified source")
        self.run_batch()
        self.assertNotEqual(self.record()["log"], second["log"])

    def test_failed_cases_require_explicit_retry(self):
        self.data["cases"].append(self.case("fail", "raise SystemExit(7)"))
        self.run_batch(1)
        failed = self.record("fail")
        passed = self.record()
        self.run_batch(1)
        self.assertEqual(self.record("fail")["log"], failed["log"])
        self.run_batch(1, "--retry-failed")
        self.assertNotEqual(self.record("fail")["log"], failed["log"])
        self.assertEqual(self.record()["log"], passed["log"])

    def test_case_change_only_invalidates_that_case(self):
        self.data["cases"].append(self.case("other", "print('other')"))
        self.run_batch()
        first, other = self.record(), self.record("other")
        self.data["cases"][0]["metadata"]["acceptance"] = "new threshold"
        self.run_batch()
        self.assertNotEqual(self.record()["log"], first["log"])
        self.assertEqual(self.record("other")["log"], other["log"])

    def test_report_helper_changes_invalidate_reuse_and_running_results(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.runner = scripts / RUNNER.name
        helper = scripts / "matrix_reports.py"
        shutil.copyfile(RUNNER, self.runner)
        shutil.copyfile(RUNNER.with_name(helper.name), helper)
        self.run_batch()
        first = self.record()
        self.run_batch()
        self.assertEqual(self.record()["log"], first["log"])
        helper.write_text(helper.read_text() + "\n")
        self.run_batch()
        self.assertNotEqual(self.record()["log"], first["log"])
        code = f"from pathlib import Path; p=Path({str(helper)!r}); p.write_text(p.read_text()+'\\n')"
        self.data["cases"] = [self.case("mutate-helper", code)]
        self.run_batch(2)
        self.assertEqual(self.record("mutate-helper")["status"], "stale")

    def test_timeout_kills_descendants(self):
        marker = self.root / "leaked-child"
        child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
        code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(20)"
        self.data["cases"] = [self.case("timeout", code, 0.2)]
        self.run_batch(1)
        self.assertEqual(self.record("timeout")["status"], "timeout")
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_mutating_source_cannot_pass(self):
        self.data["cases"] = [self.case("mutate", "from pathlib import Path; Path('source.txt').write_text('new')")]
        self.run_batch(2)
        self.assertEqual(self.record("mutate")["status"], "stale")

    def test_interrupt_and_directory_lock(self):
        marker = self.root / "started"
        code = f"from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep(20)"
        self.data["cases"] = [self.case("wait", code, 30)]
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.exists())
            result = self.run_batch(2)
            self.assertIn("Another runner owns", result.stderr)
            os.kill(process.pid, signal.SIGTERM)
            process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130)
            self.assertEqual(self.record("wait")["status"], "interrupted")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_invalid_manifest_does_not_execute(self):
        self.data["cases"][0]["id"] = "../escape"
        self.run_batch(2)
        self.assertFalse((self.root / "escape.json").exists())

    def test_artifact_validation_requires_revision_and_inputs(self):
        self.data["repositories"] = []
        self.run_batch(2)
        self.data["context"]["artifact_revision"] = "test-build"
        self.run_batch(2)
        self.data["inputs"] = [str(self.model)]
        self.run_batch()
        self.assertEqual(self.record()["provenance"]["repositories"], {})
        first = self.record()
        self.model.write_bytes(b"new artifact")
        self.run_batch()
        self.assertNotEqual(self.record()["log"], first["log"])

    def test_ci_manifest_runs_installed_suites_without_checkout(self):
        install = self.root / "install"
        tests = install / "tests"
        scripts = tests / "gguf_validation"
        scripts.mkdir(parents=True)
        for name in ("run_matrix.py", "matrix_reports.py", "make_ci_matrix.py"):
            shutil.copyfile(RUNNER.with_name(name), scripts / name)
        (scripts / "ci_build_info.json").write_text(json.dumps({"artifact_revision": "test-build"}))
        for name, count in (("ov_gguf_frontend_tests", 511), ("ov_gguf_architecture_library_tests", 1)):
            binary = tests / name
            xml = "<testsuites>" + "<testcase/>" * count + "</testsuites>"
            binary.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                              f"Path(sys.argv[-1].split('xml:',1)[1]).write_text({xml!r})\n")
            binary.chmod(0o755)
        fixture = tests / "test_data/arch_fixtures/sample.gguf.hdr"
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"fixture")
        library = install / "runtime/libopenvino.so"
        library.parent.mkdir()
        library.write_bytes(b"runtime")
        subprocess.run([sys.executable, str(scripts / "make_ci_matrix.py"), "--install-dir", str(install),
                        "--output", str(self.manifest)], check=True, capture_output=True)
        self.data = json.loads(self.manifest.read_text())
        self.runner = scripts / "run_matrix.py"
        self.assertIn(str(fixture), self.data["inputs"])
        self.assertIn(str(library), self.data["inputs"])
        self.run_batch()
        self.assertEqual(self.record("frontend")["artifacts"][0]["audit"]["tests"], 511)
        self.assertEqual(self.record("architecture-library")["artifacts"][0]["audit"]["tests"], 1)

    def test_output_setup_errors_return_two(self):
        parent = self.root / "file"
        parent.write_text("not a directory")
        loop = self.root / "loop"
        loop.symlink_to(loop)
        for output in (parent / "results", loop):
            self.output = output
            result = self.run_batch(2)
            self.assertIn("Validation setup error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def report_case(self, kind, content):
        path = self.root / ("tests.xml" if kind == "gtest" else "accuracy.json")
        code = f"from pathlib import Path; Path({str(path)!r}).write_text({content!r})"
        case = self.case("pass", code)
        contract = {"path": str(path), "kind": kind}
        if kind == "gtest":
            contract["expected_tests"] = 2
        else:
            contract.update({"required_modalities": ["text", "image"], "required_api_checks": ["reset"],
                             "min_choice_fraction": 0.9, "first_token_matches": True,
                             "equals": {"reference_revision": "pinned", "request_reset_matches": True}})
        case["reports"] = [contract]
        self.data["cases"] = [case]
        return path

    def gguf_report(self):
        return {"passed": True, "completed": True, "reference_revision": "pinned", "request_reset_matches": True,
                "cases": [{"modality": modality, "tokens": [1, 2], "reference_choices_on_same_history": [1, 2],
                           "first_token_matches": True, "matching_choice_fraction": 1.0}
                          for modality in ("text", "image")], "api_checks": {"reset": {"passed": True}}}

    def test_gtest_report_rejects_skips_and_wrong_count(self):
        for content in ("<testsuites><testcase/><testcase><skipped/></testcase></testsuites>",
                        "<testsuites><testcase/></testsuites>",
                        "<testsuites><testcase/><testcase><failure/></testcase></testsuites>"):
            self.report_case("gtest", content)
            self.run_batch(1)
            self.assertEqual(self.record()["returncode"], 0)
            self.assertTrue(self.record()["report_errors"])

    def test_explicit_skip_allowance_and_malformed_report(self):
        self.report_case("gtest", "<testsuites><testcase/><testcase><skipped/></testcase></testsuites>")
        self.data["cases"][0]["reports"][0]["max_skips"] = 1
        self.run_batch()
        self.assertEqual(self.record()["artifacts"][0]["audit"]["skips"], 1)
        self.report_case("gtest", "malformed XML")
        self.run_batch(1)
        self.assertTrue(self.record()["report_errors"])

    def test_invalid_report_contract_does_not_execute(self):
        self.report_case("gguf_mmproj", json.dumps(self.gguf_report()))
        self.data["cases"][0]["reports"][0]["min_choice_fraction"] = float("nan")
        self.run_batch(2)
        self.assertFalse((self.output / "pass.json").exists())

    def test_report_snapshot_survives_original_overwrite(self):
        path = self.report_case("gtest", "<testsuites><testcase/><testcase/></testsuites>")
        self.run_batch()
        first = self.record()
        snapshot = Path(first["artifacts"][0]["path"])
        path.write_text("overwritten by another run")
        self.run_batch()
        self.assertEqual(self.record()["log"], first["log"])
        self.assertIn("<testcase/>", snapshot.read_text())
        snapshot.write_text("corrupt")
        self.run_batch()
        self.assertNotEqual(self.record()["log"], first["log"])

    def test_missing_and_old_report_cannot_pass(self):
        path = self.report_case("gtest", "<testsuites><testcase/><testcase/></testsuites>")
        self.data["cases"][0]["command"] = [sys.executable, "-c", "print('no report')"]
        self.run_batch(1)
        path.write_text("<testsuites><testcase/><testcase/></testsuites>")
        self.run_batch(1, "--retry-failed")
        self.assertIn("unchanged", self.record()["report_errors"][0])

    def test_gguf_report_acceptance_and_recomputed_metrics(self):
        report = self.gguf_report()
        self.report_case("gguf_mmproj", json.dumps(report))
        self.run_batch()
        self.assertEqual(self.record()["artifacts"][0]["audit"]["minimum_choice_fraction"], 1)
        report["cases"][1]["reference_choices_on_same_history"][0] = 99
        self.report_case("gguf_mmproj", json.dumps(report))
        self.run_batch(1)
        self.assertIn("scores disagree", self.record()["report_errors"][0])
        report["cases"][1].update({"first_token_matches": False, "matching_choice_fraction": 0.5})
        self.report_case("gguf_mmproj", json.dumps(report))
        self.run_batch(1)
        self.assertIn("choice_fraction=0.5", self.record()["report_errors"][0])

    def test_gguf_missing_coverage_precision_and_nonfinite_metrics(self):
        for mutation in (lambda r: r["cases"].pop(),
                         lambda r: r["api_checks"].clear(),
                         lambda r: r.update(reference_revision="other"),
                         lambda r: r.update(completed=False),
                         lambda r: r["cases"][0].update(matching_choice_fraction=float("nan")),
                         lambda r: r["cases"][0].update(tokens=[]),
                         lambda r: r["api_checks"]["reset"].update(passed=False)):
            report = self.gguf_report()
            mutation(report)
            self.report_case("gguf_mmproj", json.dumps(report))
            self.run_batch(1)
            self.assertTrue(self.record()["report_errors"])

    def test_invalidation_preserves_attempt_metadata_and_summary(self):
        self.run_batch()
        first = self.record()
        old_summary = json.loads((self.output / "summary.json").read_text())
        self.model.write_bytes(b"new")
        self.run_batch()
        self.assertEqual(json.loads(Path(first["log"]).with_suffix(".json").read_text()), first)
        summaries = [json.loads(p.read_text()) for p in self.output.glob("summary-*.json")]
        self.assertIn(old_summary, summaries)

    def test_report_cannot_be_input_or_shared_by_cases(self):
        path = self.report_case("gtest", "<testsuites><testcase/><testcase/></testsuites>")
        path.write_text("existing")
        self.data["inputs"] = [str(path)]
        self.run_batch(2)
        self.data["inputs"] = []
        other = self.case("other", "print('ok')")
        other["reports"] = self.data["cases"][0]["reports"]
        self.data["cases"].append(other)
        self.run_batch(2)

    def test_failed_command_retains_report_snapshot(self):
        self.report_case("gguf_mmproj", json.dumps(self.gguf_report()))
        self.data["cases"][0]["command"][-1] += "; raise SystemExit(7)"
        self.run_batch(1)
        record = self.record()
        self.assertEqual(record["returncode"], 7)
        self.assertTrue(Path(record["artifacts"][0]["path"]).is_file())

    def test_failed_gguf_report_preserves_numerical_reason(self):
        report = self.gguf_report()
        report["passed"] = False
        report["cases"][1].update({"reference_choices_on_same_history": [99, 2],
                                   "first_token_matches": False, "matching_choice_fraction": 0.5})
        self.report_case("gguf_mmproj", json.dumps(report))
        self.run_batch(1)
        self.assertIn("GGUF image: first_token=False, choice_fraction=0.5", self.record()["report_errors"][0])
        report = self.gguf_report()
        report["passed"] = False
        self.report_case("gguf_mmproj", json.dumps(report))
        self.run_batch(1)
        self.assertIn("did not pass", self.record()["report_errors"][0])


if __name__ == "__main__":
    unittest.main()
