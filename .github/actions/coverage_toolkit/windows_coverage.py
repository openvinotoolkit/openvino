# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess  # nosec B404 - fixed tool argv, without a shell
import sys
import time
import xml.etree.ElementTree as ET  # nosec B405 - reports from local test/coverage processes

from coverage import LOGGER


OFFLINE_PATTERNS = ("*OfflineCompilationUnitTests.*", "UnavailableDeviceTests.*")
MODULE_NAMES = ("ov_npu_unit_tests.exe", "openvino_intel_npu_plugin.dll")


def parse_inventory(text: str) -> list[str]:
    tests: list[str] = []
    suite = ""
    for line in text.splitlines():
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        if not line[0].isspace() and value.endswith("."):
            suite = value
        elif line.startswith("  ") and suite:
            tests.append(suite + value)
    if not tests or len(tests) != len(set(tests)):
        raise ValueError("GoogleTest inventory is empty or contains duplicate test names")
    return tests


def select_tests(tests: list[str], lane: str) -> list[str]:
    if lane not in {"npu", "npu-offline"}:
        raise ValueError(f"Unknown Windows coverage lane: {lane}")
    enabled = [name for name in tests if not any(
        part.startswith("DISABLED_") for part in re.split(r"[./]", name)
    )]
    for pattern in OFFLINE_PATTERNS:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in enabled):
            raise ValueError(f"Offline filter no longer matches any enabled test: {pattern}")
    selected = [name for name in enabled if
                any(fnmatch.fnmatchcase(name, pattern) for pattern in OFFLINE_PATTERNS) == (lane == "npu-offline")]
    if not selected:
        raise ValueError(f"No enabled tests selected for {lane}")
    return selected


def check_test_results(path: Path, expected: list[str], lane: str) -> None:
    root = ET.parse(path).getroot()
    actual: set[str] = set()
    failures: list[str] = []
    for suite in root.findall(".//testsuite"):
        for case in suite.findall("testcase"):
            if case.get("status") == "notrun":
                continue
            name = f"{suite.attrib['name']}.{case.attrib['name']}"
            if name in actual:
                raise ValueError(f"Duplicate test result: {name}")
            actual.add(name)
            skipped = case.find("skipped")
            if skipped is not None:
                LOGGER.warning("Skipped %s: %s", name, skipped.get("message", skipped.text))
                if lane == "npu" and ".DeviceRoundtripAccuracy" in name:
                    failures.append(f"{name}: required NPU inference was skipped")
            if case.find("failure") is not None or case.find("error") is not None:
                failures.append(name)
    if actual != set(expected):
        raise ValueError(f"Test inventory mismatch: missing={sorted(set(expected) - actual)}, "
                         f"unexpected={sorted(actual - set(expected))}")
    if failures:
        raise ValueError(f"Failed tests: {failures}")


def source_path(filename: str, build_root: str, workspace: Path) -> str:
    path = PureWindowsPath(filename)
    if path.is_absolute():
        try:
            path = path.relative_to(PureWindowsPath(build_root))
        except ValueError as exc:
            raise ValueError(f"Coverage source is outside the build checkout: {filename}") from exc
    relative = path.as_posix()
    if path.drive or ".." in path.parts or not relative.startswith("src/plugins/intel_npu/"):
        raise ValueError(f"Unexpected coverage source: {filename}")
    if "/tests/" in relative or "/thirdparty/" in relative:
        raise ValueError(f"Non-production source in coverage: {filename}")
    if not (workspace / relative).is_file():
        raise ValueError(f"Coverage source does not exist in this checkout: {relative}")
    return relative


def normalize_coverage(raw: Path, output: Path, build_root: str, workspace: Path) -> None:
    root = ET.parse(raw).getroot()
    if root.tag != "coverage":
        raise ValueError("Expected a Cobertura coverage report")
    packages = root.findall("./packages/package")
    for module in MODULE_NAMES:
        if not any(Path(module).stem.lower() in package.get("name", "").lower() and package.findall(".//line")
                   for package in packages):
            raise ValueError(f"No coverage for required module: {module}")

    # The EXE and plugin compile some of the same sources. Count each source line once.
    files: dict[str, dict[int, int]] = {}
    for cls in root.findall(".//class"):
        filename = source_path(cls.attrib["filename"], build_root, workspace)
        lines = files.setdefault(filename, {})
        for line in cls.findall("./lines/line"):
            number, hits = int(line.attrib["number"]), int(line.attrib["hits"])
            if number <= 0 or hits < 0:
                raise ValueError(f"Invalid line counter in {filename}: {line.attrib}")
            lines[number] = max(lines.get(number, 0), hits)
    total = sum(len(lines) for lines in files.values())
    covered = sum(hits > 0 for lines in files.values() for hits in lines.values())
    if not total or not covered:
        raise ValueError("Coverage has no executable or covered NPU source lines")
    attributes = {
        "line-rate": str(covered / total), "branch-rate": "0",
        "lines-valid": str(total), "lines-covered": str(covered),
        "branches-valid": "0", "branches-covered": "0", "version": "1",
    }
    normalized = ET.Element("coverage", attributes)
    ET.SubElement(ET.SubElement(normalized, "sources"), "source").text = "."
    package = ET.SubElement(ET.SubElement(normalized, "packages"), "package",
                            {"name": "windows", "line-rate": attributes["line-rate"], "branch-rate": "0"})
    classes = ET.SubElement(package, "classes")
    for filename, lines in sorted(files.items()):
        if not lines:
            continue
        cls = ET.SubElement(classes, "class", {
            "name": filename, "filename": filename,
            "line-rate": str(sum(hits > 0 for hits in lines.values()) / len(lines)), "branch-rate": "0",
        })
        ET.SubElement(cls, "methods")
        line_nodes = ET.SubElement(cls, "lines")
        for number, hits in sorted(lines.items()):
            ET.SubElement(line_nodes, "line", {"number": str(number), "hits": str(hits)})
    ET.ElementTree(normalized).write(output, encoding="utf-8", xml_declaration=True)
    LOGGER.info("Windows NPU line coverage: %d/%d (%.2f%%)", covered, total, 100 * covered / total)


def write_settings(path: Path, modules: list[Path]) -> None:
    root = ET.Element("Configuration")
    ET.SubElement(root, "IncludeTestAssembly").text = "True"
    ET.SubElement(root, "ExcludeAssembliesWithoutSources").text = "None"
    coverage = ET.SubElement(root, "CodeCoverage")
    for name, value in {
        "EnableStaticNativeInstrumentation": "True",
        "EnableDynamicNativeInstrumentation": "False",
        "EnableStaticNativeInstrumentationRestore": "True",
        "EnableStaticManagedInstrumentation": "False",
        "EnableDynamicManagedInstrumentation": "False",
    }.items():
        ET.SubElement(coverage, name).text = value
    paths = ET.SubElement(coverage, "ModulePaths")
    includes = ET.SubElement(paths, "Include")
    directories = ET.SubElement(paths, "IncludeDirectories")
    for module in modules:
        ET.SubElement(includes, "ModulePath").text = re.escape(str(module))
    for directory in sorted({module.parent for module in modules}):
        ET.SubElement(directories, "Directory", {"Recursive": "False"}).text = str(directory)
    sources = ET.SubElement(coverage, "Sources")
    ET.SubElement(ET.SubElement(sources, "Include"), "Source").text = r".*[\\/]src[\\/]plugins[\\/]intel_npu[\\/].*"
    excludes = ET.SubElement(sources, "Exclude")
    for pattern in (r".*[\\/]tests[\\/].*", r".*[\\/]thirdparty[\\/].*"):
        ET.SubElement(excludes, "Source").text = pattern
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def run_logged(command: list[str], log: Path, *, timeout: int = 600, env: dict[str, str] | None = None) -> int:
    LOGGER.info("$ %s", subprocess.list2cmdline(command))
    with log.open("w", encoding="utf-8") as handle:
        with subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env) as process:
            try:
                result = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                LOGGER.error("Timed out after %s seconds; terminating PID %s and its children", timeout, process.pid)
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
                else:
                    process.kill()
                process.wait()
                raise
    LOGGER.info("%s", log.read_text(encoding="utf-8", errors="replace"))
    return result


def check_device(install: Path, lane: str) -> None:
    import openvino as ov

    if not Path(ov.__file__).resolve().is_relative_to(install):
        raise RuntimeError(f"OpenVINO was not imported from the coverage package: {ov.__file__}")
    core = ov.Core()
    devices = core.available_devices
    has_npu = any(device.split(".")[0] == "NPU" for device in devices)
    if has_npu != (lane == "npu"):
        raise RuntimeError(f"Wrong device environment for {lane}: {devices}")
    if "CPU" not in devices:
        raise RuntimeError("CPU plugin is required by the NPUW unit tests")
    print(json.dumps({"openvino": ov.__file__, "version": ov.__version__, "devices": devices}))
    if has_npu:
        print(core.get_versions("NPU"))


def collect(workspace: Path, install: Path, collector: Path, lane: str) -> None:
    diagnostics = workspace / ".tmp" / "windows-coverage"
    diagnostics.mkdir(parents=True, exist_ok=True)
    output = workspace / "windows-coverage.xml"
    output.unlink(missing_ok=True)
    identity = json.loads((install / "coverage-build.json").read_text(encoding="utf-8-sig"))
    commit = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
    if identity["commit"] != commit or identity["build_type"] != "Release" or identity["toolchain"] != "vs2022":
        raise ValueError(f"Coverage build does not match this checkout/configuration: {identity}")
    modules = []
    for name in MODULE_NAMES:
        candidates = list(install.rglob(name))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one {name}, found {candidates}")
        module = candidates[0]
        if not module.with_suffix(".pdb").is_file():
            raise FileNotFoundError(f"Missing linker PDB for {module}")
        modules.append(module)
    if not collector.is_file():
        raise FileNotFoundError(f"Microsoft.CodeCoverage.Console is not installed: {collector}")
    if run_logged([str(collector), "--version"], diagnostics / "collector-version.log"):
        raise RuntimeError("Cannot run Microsoft.CodeCoverage.Console")
    # Use a separate process so the DLLs are unloaded before static instrumentation.
    if run_logged([sys.executable, __file__, "--install-dir", str(install), "--lane", lane, "--check-device"],
                  diagnostics / "device.log"):
        raise RuntimeError("Windows device/package preflight failed")
    executable = modules[0]
    env = {name: value for name, value in os.environ.items() if not name.startswith("GTEST_")}
    env["OV_DQ_TEST_DEVICE"] = "NPU" if lane == "npu" else ""
    inventory_log = diagnostics / "inventory.log"
    if run_logged([str(executable), "--gtest_list_tests"], inventory_log, env=env):
        raise RuntimeError("Cannot enumerate NPU unit tests")
    inventory = parse_inventory(inventory_log.read_text(encoding="utf-8", errors="replace"))
    selected = select_tests(inventory, lane)
    (diagnostics / "inventory.json").write_text(
        json.dumps({"all": inventory, "selected": selected, "lane": lane}, indent=2), encoding="utf-8")
    settings = diagnostics / "coverage.config"
    write_settings(settings, modules)
    test_xml = diagnostics / "TEST-NPUUnitTests.xml"
    test_xml.unlink(missing_ok=True)
    raw = diagnostics / "windows.coverage"
    raw.unlink(missing_ok=True)
    filter_value = ":".join(OFFLINE_PATTERNS)
    if lane == "npu":
        filter_value = "-" + filter_value
    started = time.monotonic()
    result = run_logged([
        str(collector), "collect", "--settings", str(settings),
        "--output", str(raw), "--output-format", "coverage",
        "--log-file", str(diagnostics / "collector.log"), "--log-level", "Verbose",
        str(executable), f"--gtest_filter={filter_value}", f"--gtest_output=xml:{test_xml}",
        "--gtest_print_time=1",
    ], diagnostics / "tests.log", timeout=7200, env=env)
    duration = time.monotonic() - started
    # Export before propagating test failures, so diagnostics and partial coverage survive.
    raw_xml = diagnostics / "raw-coverage.xml"
    if run_logged([str(collector), "merge", str(raw), "--output", str(raw_xml), "--output-format", "cobertura"],
                  diagnostics / "export.log"):
        raise RuntimeError("Microsoft coverage export failed")
    normalize_coverage(raw_xml, output, identity["source_root"], workspace)
    check_test_results(test_xml, selected, lane)
    if result:
        raise RuntimeError(f"Microsoft coverage/test process exited with {result}")
    with (workspace / "windows-test-durations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_name", "status", "duration_seconds", "duration_minutes"])
        writer.writerow(["ov_npu_unit_tests", "passed", f"{duration:.3f}", f"{duration / 60:.3f}"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Windows NPU coverage with Microsoft.CodeCoverage.Console")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--collector", type=Path)
    parser.add_argument("--lane", choices=("npu", "npu-offline"), required=True)
    parser.add_argument("--check-device", action="store_true")
    args = parser.parse_args()
    if args.check_device:
        check_device(args.install_dir.resolve(), args.lane)
        return 0
    if args.collector is None:
        parser.error("--collector is required for collection")
    workspace = args.workspace.resolve()
    status = "failed"
    started = time.monotonic()
    try:
        collect(workspace, args.install_dir.resolve(), args.collector.resolve(), args.lane)
        status = "passed"
    finally:
        executed = (workspace / ".tmp/windows-coverage/tests.log").is_file()
        (workspace / "windows-coverage-stats.env").write_text(
            f"CXX_TESTS_TOTAL=1\nCXX_TESTS_EXECUTED={int(executed)}\n"
            f"CXX_TESTS_PASSED={int(status == 'passed')}\nCXX_TESTS_FAILED={int(status == 'failed' and executed)}\n"
            f"CXX_TESTS_SKIPPED=0\nCXX_TESTS_NOT_RUN={int(not executed)}\n", encoding="utf-8")
        if status == "failed":
            duration = time.monotonic() - started
            with (workspace / "windows-test-durations.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["test_name", "status", "duration_seconds", "duration_minutes"])
                writer.writerow(["ov_npu_unit_tests", status if executed else "not_run",
                                 f"{duration:.3f}", f"{duration / 60:.3f}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
