# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import ci_reports
from coverage import validate_configs
import windows_coverage as windows


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / ".github" / "coverage"
INVENTORY = """Running main() from gtest_main.cc
OfflineCompilationPlatforms/OfflineCompilationUnitTests.
  CompileWithCiPWhenDriverNotInstalled/NPU5010  # GetParam() = ...
UnavailableDeviceTests.
  GetDeviceNotAvailable
NPUWTests.
  Regular
  DISABLED_Unsupported
DeviceTests/DynamicQuantizeAccuracyTest.
  DeviceRoundtripAccuracy/0
"""


class WindowsCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.source = self.workspace / "src/plugins/intel_npu/src/example.cpp"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("first\nsecond\n", encoding="utf-8")
        self.env = patch.dict(os.environ, {"COVERAGE_CONFIG_DIR": str(CONFIG)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_disjoint_complete_partition(self):
        tests = windows.parse_inventory(INVENTORY)
        hardware = windows.select_tests(tests, "npu")
        offline = windows.select_tests(tests, "npu-offline")
        self.assertEqual(len(hardware), 2)
        self.assertEqual(len(offline), 2)
        self.assertFalse(set(hardware) & set(offline))
        self.assertEqual(set(hardware + offline), {name for name in tests if "DISABLED_" not in name})

    def test_empty_duplicate_or_stale_inventory_fails(self):
        for text in ("", "Tests.\n  A\n  A\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                windows.parse_inventory(text)
        with self.assertRaisesRegex(ValueError, "Offline filter"):
            windows.select_tests(["Regular.Test"], "npu")

    def test_windows_source_mapping(self):
        self.assertEqual(windows.source_path(
            r"C:\Build\OpenVINO\src\plugins\intel_npu\src\example.cpp",
            r"c:\build\openvino", self.workspace),
            "src/plugins/intel_npu/src/example.cpp")
        for path in (r"D:\unrelated.cpp", "../escape.cpp", r"C:relative.cpp",
                     "src/plugins/intel_npu/tests/test.cpp", "src/plugins/intel_npu/src/missing.cpp"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                windows.source_path(path, r"C:\Build\OpenVINO", self.workspace)

    def report(self, modules=windows.MODULE_NAMES, hits=1):
        root = ET.Element("coverage")
        packages = ET.SubElement(root, "packages")
        for module in modules:
            package = ET.SubElement(packages, "package", {"name": module})
            cls = ET.SubElement(ET.SubElement(package, "classes"), "class", {
                "filename": r"C:\build\openvino\src\plugins\intel_npu\src\example.cpp",
            })
            lines = ET.SubElement(cls, "lines")
            ET.SubElement(lines, "line", {"number": "1", "hits": str(hits)})
            ET.SubElement(lines, "line", {"number": "2", "hits": "0"})
        raw = self.workspace / "raw.xml"
        ET.ElementTree(root).write(raw)
        return raw

    def test_report_preserves_uncovered_lines_and_merges_duplicate_sources(self):
        output = self.workspace / "windows-coverage.xml"
        windows.normalize_coverage(self.report(), output, r"C:\build\openvino", self.workspace)
        root = ET.parse(output).getroot()
        self.assertEqual(root.get("lines-valid"), "2")
        self.assertEqual(root.get("lines-covered"), "1")
        self.assertEqual(root.find(".//class").get("filename"), "src/plugins/intel_npu/src/example.cpp")
        self.assertEqual([line.get("hits") for line in root.findall(".//line")], ["1", "0"])

    def test_empty_coverage_or_missing_module_fails(self):
        for modules, hits in ((windows.MODULE_NAMES[:1], 1), (windows.MODULE_NAMES, 0)):
            with self.subTest(modules=modules, hits=hits), self.assertRaises(ValueError):
                windows.normalize_coverage(self.report(modules, hits), self.workspace / "out.xml",
                                           r"C:\build\openvino", self.workspace)

    def test_native_settings_include_exe_and_dll_without_dynamic_fallback(self):
        path = self.workspace / "coverage.config"
        windows.write_settings(path, [self.workspace / name for name in windows.MODULE_NAMES])
        root = ET.parse(path).getroot()
        self.assertEqual(root.findtext("IncludeTestAssembly"), "True")
        self.assertEqual(root.findtext("CodeCoverage/EnableStaticNativeInstrumentation"), "True")
        self.assertEqual(root.findtext("CodeCoverage/EnableDynamicNativeInstrumentation"), "False")
        self.assertEqual(len(root.findall("CodeCoverage/ModulePaths/Include/ModulePath")), 2)

    def test_gtest_missing_failed_or_required_skipped_results_fail(self):
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite", {"name": "DeviceTests"})
        case = ET.SubElement(suite, "testcase", {"name": "DeviceRoundtripAccuracy", "status": "run"})
        path = self.workspace / "tests.xml"
        ET.ElementTree(root).write(path)
        expected = ["DeviceTests.DeviceRoundtripAccuracy"]
        windows.check_test_results(path, expected, "npu")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            windows.check_test_results(path, expected + ["Missing.Test"], "npu")
        for tag in ("failure", "skipped"):
            child = ET.SubElement(case, tag)
            ET.ElementTree(root).write(path)
            with self.assertRaisesRegex(ValueError, "Failed tests"):
                windows.check_test_results(path, expected, "npu")
            case.remove(child)

    def test_missing_symbols_fail_before_execution(self):
        install = self.workspace / "install"
        install.mkdir()
        (install / "coverage-build.json").write_text(json.dumps({
            "commit": "abc", "build_type": "Release", "toolchain": "vs2026",
        }))
        (install / windows.MODULE_NAMES[0]).touch()
        with patch.object(subprocess, "check_output", return_value="abc\n"), \
                self.assertRaisesRegex(FileNotFoundError, "Missing linker PDB"):
            windows.collect(self.workspace, install, self.workspace / "collector.exe", "npu")

    def test_windows_artifacts_and_uploads_are_isolated(self):
        output = self.workspace / "windows-coverage.xml"
        windows.normalize_coverage(self.report(), output, r"C:\build\openvino", self.workspace)
        artifact_name, artifact = ci_reports.collect_suite_results(
            workspace=self.workspace, suite="windows", profile="npu", lane="npu", test_names="",
            artifact_name="", artifact_dir=self.workspace / "artifacts/windows/coverage-windows-npu",
            outputs_file=None,
        )
        self.assertEqual(artifact_name, "coverage-windows-npu")
        metadata = json.loads((artifact / ci_reports.METADATA_FILE).read_text())
        self.assertEqual(metadata["suite"], "windows")
        matrix = ci_reports._build_upload_matrix(workspace=self.workspace)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["flag"], "cpp-runtime-cpp-windows-npu")
        self.assertEqual(matrix[0]["file"], "windows-coverage.xml")
        self.assertEqual(matrix[0]["artifact_group"], "windows")
        self.assertEqual(ci_reports._collect_artifacts(workspace=self.workspace, suite_key="cpp"), [])

    def test_workflow_and_codecov_contracts(self):
        import yaml

        workflow = yaml.safe_load((REPO / ".github/workflows/coverage.yml").read_text())
        jobs = workflow["jobs"]
        build = jobs["BuildWindows"]["with"]
        self.assertEqual(build["build-variant"], "vs2026")
        self.assertTrue(build["enable-coverage"])
        self.assertNotIn("needs", jobs["BuildWindows"])
        lanes = jobs["CoverageWindows"]["strategy"]["matrix"]["include"]
        self.assertEqual({entry["lane"] for entry in lanes}, {"npu", "npu-offline"})
        self.assertIn("BuildWindows", jobs["CoverageSummary"]["needs"])
        codecov = yaml.safe_load((REPO / ".github/codecov.yml").read_text())
        for kind in ("project", "patch"):
            statuses = codecov["coverage"]["status"][kind]
            self.assertFalse(statuses["default"])
            self.assertFalse(set(statuses["linux"]["flags"]) & set(statuses["windows"]["flags"]))
            self.assertEqual(set(statuses["windows"]["flags"]),
                             {"cpp-runtime-cpp-windows-npu", "cpp-runtime-cpp-windows-npu-offline"})
        self.assertFalse(codecov["flag_management"]["default_rules"]["carryforward"])
        self.assertEqual(validate_configs(CONFIG), [])

    @unittest.skipUnless(shutil.which("bash"), "bash is required to validate lane selection")
    def test_pull_request_and_dispatch_selection(self):
        import yaml

        workflow = yaml.safe_load((REPO / ".github/workflows/coverage.yml").read_text())
        # PyYAML's YAML 1.1 loader interprets the unquoted "on" key as True.
        self.assertIn("pull_request", workflow[True])
        jobs = workflow["jobs"]
        schedule = "github.event_name == 'schedule'"
        cpu = "github.event.inputs.run-cpu-tests == 'true'"
        gpu = "github.event.inputs.run-gpu-tests == 'true'"
        for job in ("Docker", "Build"):
            self.assertEqual(jobs[job]["if"], "${{ " + f"{schedule} || {cpu} || {gpu}" + " }}")
        for job, toggle in (("CoverageCpp", cpu), ("CoveragePython", cpu), ("CoverageJS", cpu),
                            ("CoverageCppGpu", gpu), ("CoveragePythonGpu", gpu)):
            self.assertEqual(jobs[job]["if"], "${{ " + f"{schedule} || {toggle}" + " }}")
        for job in ("BuildWindows", "CoverageWindows"):
            self.assertEqual(jobs[job]["if"],
                             "${{ github.event_name == 'schedule' || "
                             "github.event_name == 'pull_request' || inputs.run-npu-tests }}")
        script = next(step["run"] for step in jobs["CoverageSummary"]["steps"] if step.get("id") == "lanes")
        linux = ["cpu", "igpu", "igpu-unit", "igpu-func", "dgpu", "dgpu-unit", "dgpu-func"]
        npu = ["npu", "npu-offline"]
        cases = [("pull_request", "", "", "", npu), ("schedule", "", "", "", linux + npu)]
        for run_cpu, run_gpu, run_npu in itertools.product(("false", "true"), repeat=3):
            expected = (linux[:1] if run_cpu == "true" else []) + (linux[1:] if run_gpu == "true" else [])
            expected += npu if run_npu == "true" else []
            cases.append(("workflow_dispatch", run_cpu, run_gpu, run_npu, expected))
        output = self.workspace / "github-output"
        for event, run_cpu, run_gpu, run_npu, expected in cases:
            with self.subTest(event=event, cpu=run_cpu, gpu=run_gpu, npu=run_npu):
                output.write_text("")
                subprocess.run(["bash", "-e", "-c", script], check=True, env={
                    **os.environ, "EVENT_NAME": event, "RUN_CPU_TESTS": run_cpu,
                    "RUN_GPU_TESTS": run_gpu, "RUN_NPU_TESTS": run_npu, "GITHUB_OUTPUT": str(output),
                })
                self.assertEqual(output.read_text().strip(), "selected_lanes=" + (",".join(expected) or "none"))

    def test_collection_orchestration(self):
        install = self.workspace / "install"
        install.mkdir()
        (install / "coverage-build.json").write_text(json.dumps({
            "commit": "abc", "build_type": "Release", "toolchain": "vs2026",
            "source_root": r"C:\build\openvino",
        }))
        for module in windows.MODULE_NAMES:
            (install / module).touch()
            (install / module).with_suffix(".pdb").touch()
        collector = self.workspace / "Microsoft.CodeCoverage.Console.exe"
        collector.touch()
        raw = self.report()
        commands = []

        def run(command, log, **kwargs):
            commands.append(command)
            if "--gtest_list_tests" in command:
                log.write_text(INVENTORY)
            elif "collect" in command:
                self.assertEqual(kwargs["env"]["OV_DQ_TEST_DEVICE"], "NPU")
                root = ET.Element("testsuites")
                for name in windows.select_tests(windows.parse_inventory(INVENTORY), "npu"):
                    suite_name, test_name = name.rsplit(".", 1)
                    suite = ET.SubElement(root, "testsuite", {"name": suite_name})
                    ET.SubElement(suite, "testcase", {"name": test_name, "status": "run"})
                xml_path = next(arg.removeprefix("--gtest_output=xml:") for arg in command
                                if arg.startswith("--gtest_output=xml:"))
                ET.ElementTree(root).write(xml_path)
            elif "merge" in command:
                Path(command[command.index("--output") + 1]).write_bytes(raw.read_bytes())
            return 0

        with patch.object(subprocess, "check_output", return_value="abc\n"), \
                patch.object(windows, "run_logged", side_effect=run):
            windows.collect(self.workspace, install, collector, "npu")
        self.assertTrue((self.workspace / "windows-coverage.xml").is_file())
        self.assertTrue((self.workspace / "windows-test-durations.csv").is_file())
        self.assertTrue(any("--check-device" in command for command in commands))

    def test_preflight_failure_is_reported_as_not_run(self):
        args = ["windows_coverage.py", "--workspace", str(self.workspace),
                "--install-dir", str(self.workspace / "install"),
                "--collector", str(self.workspace / "collector.exe"), "--lane", "npu"]
        with patch("sys.argv", args), patch.object(windows, "collect", side_effect=RuntimeError("preflight")), \
                self.assertRaisesRegex(RuntimeError, "preflight"):
            windows.main()
        stats = (self.workspace / "windows-coverage-stats.env").read_text()
        self.assertIn("CXX_TESTS_EXECUTED=0", stats)
        self.assertIn("CXX_TESTS_NOT_RUN=1", stats)
        self.assertIn("CXX_TESTS_PASSED=0", stats)

    def test_cmake_coverage_flags(self):
        if shutil.which("cmake") is None:
            self.skipTest("CMake is not installed")
        for compiler, enabled in (("MSVC", "OFF"), ("MSVC", "ON"), ("GNU", "ON")):
            with self.subTest(compiler=compiler, coverage=enabled):
                source = self.workspace / f"cmake-{compiler}-{enabled}"
                source.mkdir()
                (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(CoverageFlags CXX)
list(APPEND CMAKE_MODULE_PATH "{REPO.as_posix()}/cmake/developer_package")
set(CMAKE_CXX_COMPILER_ID {compiler})
set(CMAKE_CXX_COMPILER_VERSION 19.50)
set(CMAKE_COMPILER_IS_GNUCXX {'ON' if compiler == 'GNU' else 'OFF'})
set(CMAKE_CL_64 {'ON' if compiler == 'MSVC' else 'OFF'})
set(ENV{{GITHUB_ACTIONS}} true)
set(ENABLE_COVERAGE {enabled})
set(SUGGEST_OVERRIDE_SUPPORTED OFF CACHE BOOL "")
set(UNUSED_BUT_SET_VARIABLE_SUPPORTED OFF CACHE BOOL "")
include(compile_flags/os_flags)
file(WRITE "${{CMAKE_BINARY_DIR}}/flags.txt"
    "${{CMAKE_CXX_FLAGS}}\\n${{CMAKE_EXE_LINKER_FLAGS}}\\n${{CMAKE_EXE_LINKER_FLAGS_RELEASE}}")
""", encoding="utf-8")
                result = subprocess.run(["cmake", "-S", str(source), "-B", str(source / "build")],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                flags = (source / "build/flags.txt").read_text()
                if compiler == "GNU":
                    self.assertIn("--coverage", flags)
                    self.assertNotIn("/PROFILE", flags)
                elif enabled == "ON":
                    for flag in ("/Zi", "/PROFILE", "/DEBUG:FULL", "/OPT:NOICF", "/INCREMENTAL:NO"):
                        self.assertIn(flag, flags)
                    self.assertNotIn("--coverage", flags)
                    self.assertNotIn("/OPT:ICF", flags)
                else:
                    self.assertIn("/OPT:ICF", flags)
                    self.assertNotIn("/PROFILE", flags)

    def test_powershell_syntax(self):
        if shutil.which("pwsh") is None:
            self.skipTest("PowerShell is not installed")
        import re
        import yaml

        scripts = []
        for filename in ("coverage.yml", "job_build_windows.yml"):
            workflow = yaml.safe_load((REPO / ".github/workflows" / filename).read_text())
            for job in workflow["jobs"].values():
                default_shell = job.get("defaults", {}).get("run", {}).get("shell")
                for step in job.get("steps", []):
                    if "run" in step and step.get("shell", default_shell) == "pwsh":
                        scripts.append({"name": f"{filename}: {step['name']}",
                                        "text": re.sub(r"\$\{\{.*?\}\}", "VALUE", step["run"])})
        command = """
$scripts = [Console]::In.ReadToEnd() | ConvertFrom-Json
$failed = $false
foreach ($script in $scripts) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseInput($script.text, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors.Count) {
        Write-Output $script.name
        Write-Output $errors
        $failed = $true
    }
}
if ($failed) { exit 1 }
"""
        result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
                                input=json.dumps(scripts), capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
