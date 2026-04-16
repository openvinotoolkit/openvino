# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from coverage_workflow import (
    CoverageContext,
    CppTestCase,
    LOGGER,
    env_from_assignments,
    load_cpp_tests,
    load_js_tests,
    load_python_tests,
    run_cmd,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "config"


@dataclass(frozen=True)
class _SuiteSpec:
    label: str
    selection_env: str
    duration_file: str
    duration_env_key: str
    duration_total_env_key: str | None
    stats_file: str
    total_key: str
    executed_key: str | None
    passed_key: str
    failed_key: str
    skipped_key: str
    not_run_key: str


@dataclass(frozen=True)
class _TestRunResult:
    name: str
    status: str
    detail: str = ""
    duration_seconds: float = 0.0


CPP_SUITE = _SuiteSpec(
    label="C++",
    selection_env="CXX_TEST_NAMES",
    duration_file="cpp-test-durations.csv",
    duration_env_key="CXX_TEST_DURATION_REPORT",
    duration_total_env_key="CXX_TEST_DURATION_TOTAL_SECONDS",
    stats_file="cpp-coverage-stats.env",
    total_key="CXX_TESTS_TOTAL",
    executed_key="CXX_TESTS_EXECUTED",
    passed_key="CXX_TESTS_PASSED",
    failed_key="CXX_TESTS_FAILED",
    skipped_key="CXX_TESTS_SKIPPED",
    not_run_key="CXX_TESTS_NOT_RUN",
)

PYTHON_SUITE = _SuiteSpec(
    label="Python",
    selection_env="PY_TEST_NAMES",
    duration_file="python-test-durations.csv",
    duration_env_key="PY_TEST_DURATION_REPORT",
    duration_total_env_key=None,
    stats_file="python-coverage-stats.env",
    total_key="PY_TESTS_TOTAL",
    executed_key=None,
    passed_key="PY_TESTS_PASSED",
    failed_key="PY_TESTS_FAILED",
    skipped_key="PY_TESTS_SKIPPED",
    not_run_key="PY_TESTS_NOT_RUN",
)

JS_SUITE = _SuiteSpec(
    label="JS",
    selection_env="JS_TEST_NAMES",
    duration_file="js-test-durations.csv",
    duration_env_key="JS_TEST_DURATION_REPORT",
    duration_total_env_key=None,
    stats_file="js-coverage-stats.env",
    total_key="JS_TESTS_TOTAL",
    executed_key=None,
    passed_key="JS_TESTS_PASSED",
    failed_key="JS_TESTS_FAILED",
    skipped_key="JS_TESTS_SKIPPED",
    not_run_key="JS_TESTS_NOT_RUN",
)


def _runtime_ld_library_path(ctx: CoverageContext, *, extra_paths: tuple[Path, ...] = ()) -> str:
    """Build the runtime library search path used by test binaries."""
    paths: list[Path] = []
    candidates = [
        ctx.paths.bin_dir,
        ctx.paths.install_pkg_dir / "runtime" / "lib" / "intel64",
        ctx.paths.install_pkg_dir / "runtime" / "3rdparty" / "tbb" / "lib",
        *extra_paths,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)

    tbb_dirs = sorted({p.parent for p in ctx.paths.install_pkg_dir.rglob("libtbb.so*") if p.is_file()})
    for tbb_dir in tbb_dirs:
        if tbb_dir not in paths:
            paths.append(tbb_dir)

    return ":".join(str(path) for path in paths)


def _selected_test_names(env_name: str) -> list[str]:
    """Read the optional test selection list for one suite."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def _filter_selected_tests(tests: list[object], selected_names: list[str], *, suite_label: str) -> list[object]:
    """Filter configured tests by requested names and warn on unknown entries."""
    if not selected_names:
        return tests

    selected_set = set(selected_names)
    known_names = {getattr(test, "name") for test in tests}
    missing_names = [name for name in selected_names if name not in known_names]
    if missing_names:
        LOGGER.warning("Requested %s tests were not found in config: %s", suite_label, ", ".join(missing_names))
    return [test for test in tests if getattr(test, "name") in selected_set]


def _timed_run(
    title: str,
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, float]:
    """Run one command and return its exit code and duration."""
    LOGGER.info("========== Running %s ==========", title)
    started_at = time.monotonic()
    rc = run_cmd(cmd, cwd=cwd, env=env, check=False)
    return rc, time.monotonic() - started_at


def _write_duration_report(ctx: CoverageContext, suite: _SuiteSpec, results: list[_TestRunResult]) -> None:
    """Write per-test durations for one suite."""
    report_path = ctx.workspace / suite.duration_file
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_name", "status", "duration_seconds", "duration_minutes"])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.status,
                    f"{result.duration_seconds:.3f}",
                    f"{result.duration_seconds / 60.0:.3f}",
                ]
            )

    ctx.io.export_env(suite.duration_env_key, str(report_path))
    if suite.duration_total_env_key:
        total_duration = sum(result.duration_seconds for result in results)
        ctx.io.export_env(suite.duration_total_env_key, f"{total_duration:.3f}")


def _write_stats_report(
    ctx: CoverageContext,
    suite: _SuiteSpec,
    *,
    total: int,
    executed: int,
    passed: int,
    failed: int,
    skipped: int,
    not_run: int = 0,
) -> None:
    """Write aggregate suite counters and export them to the workflow env."""
    values = {
        suite.total_key: total,
        suite.passed_key: passed,
        suite.failed_key: failed,
        suite.skipped_key: skipped,
        suite.not_run_key: not_run,
    }
    if suite.executed_key:
        values[suite.executed_key] = executed

    for key, value in values.items():
        ctx.io.export_env(key, str(value))

    lines = [f"{key}={value}" for key, value in values.items()]
    (ctx.workspace / suite.stats_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_summary(
    ctx: CoverageContext,
    suite: _SuiteSpec,
    *,
    selected_names: list[str],
    executed: int,
    passed: int,
    failed: list[str],
    skipped: list[str],
    extra_lines: list[str] | None = None,
) -> None:
    """Append a human-readable step summary for one suite."""
    lines = [f"## {suite.label} coverage test execution summary", ""]
    if extra_lines:
        lines.extend(extra_lines)
    lines.extend(
        [
            f"{suite.label} test selection: {', '.join(selected_names) if selected_names else 'all configured tests'}",
            f"{suite.label} tests executed: {executed}",
            f"{suite.label} tests passed: {passed}",
            f"{suite.label} tests failed: {len(failed)}",
            f"{suite.label} tests skipped: {len(skipped)}",
            "",
        ]
    )

    if failed:
        lines.append("Failed tests:")
        lines.extend(f"- {item}" for item in failed)
    else:
        lines.append("Failed tests: none")

    lines.append("")

    if skipped:
        lines.append("Skipped tests:")
        lines.extend(f"- {item}" for item in skipped)
    else:
        lines.append("Skipped tests: none")

    ctx.io.append_summary("\n".join(lines) + "\n")


def _finalize_suite(
    ctx: CoverageContext,
    suite: _SuiteSpec,
    *,
    selected_names: list[str],
    results: list[_TestRunResult],
    extra_lines: list[str] | None = None,
    no_execution_warning: str | None = None,
    failure_warning: str | None = None,
) -> None:
    """Write reports, export counters, and append summary for one suite."""
    executed = [result for result in results if result.status != "skipped"]
    failed = [result.detail for result in results if result.status == "failed" and result.detail]
    skipped = [result.detail for result in results if result.status == "skipped" and result.detail]
    passed = [result for result in results if result.status == "passed"]

    _write_duration_report(ctx, suite, results)
    _write_stats_report(
        ctx,
        suite,
        total=len(results),
        executed=len(executed),
        passed=len(passed),
        failed=len(failed),
        skipped=len(skipped),
        not_run=0,
    )
    _append_summary(
        ctx,
        suite,
        selected_names=selected_names,
        executed=len(executed),
        passed=len(passed),
        failed=failed,
        skipped=skipped,
        extra_lines=extra_lines,
    )

    if no_execution_warning and not executed:
        LOGGER.warning("%s", no_execution_warning)
    if failure_warning and failed:
        LOGGER.warning("%s", failure_warning)


def _pycov_config(
    *,
    branch_coverage: bool,
    runtime_source_dir: Path,
    repo_source_dir: Path,
    workspace: Path,
    installed_dirs: tuple[Path, ...] = (),
) -> str:
    """Build the coverage.py config used by pytest-cov and coverage CLI."""
    branch_line = "branch = True\n" if branch_coverage else ""
    try:
        source_value = runtime_source_dir.relative_to(workspace).as_posix()
    except ValueError:
        source_value = runtime_source_dir.as_posix()

    path_entries: list[str] = []
    for path in (repo_source_dir, runtime_source_dir, *installed_dirs):
        try:
            value = path.relative_to(workspace).as_posix()
        except ValueError:
            value = path.as_posix()
        if value not in path_entries:
            path_entries.append(value)
    path_entries_text = "\n    ".join(path_entries)
    return f"""[run]
source =
    {source_value}
{branch_line}omit =
    */tests/*
    */thirdparty/*
    */docs/*
    */samples/*
    */tools/*
    */src/bindings/js/node/tests/*
    */src/bindings/python/tests/*
    *.pb.cc
    *.pb.h

[paths]
openvino =
    {path_entries_text}
"""


def _expand(value: str) -> str:
    """Expand environment variables in a config value."""
    return os.path.expandvars(value)


def _remove_gcda(root: Path) -> None:
    """Delete stale gcda files before a fresh coverage run."""
    if not root.exists():
        return
    for gcda in root.rglob("*.gcda"):
        try:
            gcda.unlink()
        except OSError:
            pass


def _python_pytest_command(target: str, args: str, *, py_cov_source: str, py_cov_config: Path) -> list[str]:
    """Build one pytest command with shared Python coverage arguments."""
    cmd = ["python3", "-m", "pytest", "-ra", "--durations=50", target]
    if args:
        cmd.extend(shlex.split(args))
    cmd.extend(
        [
            f"--cov={py_cov_source}",
            f"--cov-config={py_cov_config}",
            "--cov-append",
        ]
    )
    return cmd


def _detect_installed_python_package_dir(package: str) -> Path | None:
    """Resolve the installed package directory used by the active Python interpreter."""
    cmd = [
        "python3",
        "-c",
        (
            "import importlib, pathlib; "
            f"module = importlib.import_module({package!r}); "
            "print(pathlib.Path(module.__file__).resolve().parent)"
        ),
    ]
    display = " ".join(shlex.quote(part) for part in cmd)
    LOGGER.info("$ %s", display)
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
        LOGGER.warning("Could not resolve installed Python package dir for %s: %s", package, details)
        return None

    resolved_text = completed.stdout.strip().splitlines()
    if not resolved_text:
        LOGGER.warning("Could not resolve installed Python package dir for %s: empty output", package)
        return None

    package_dir = Path(resolved_text[-1].strip())
    if not package_dir.is_dir():
        LOGGER.warning("Resolved Python package dir for %s is not a directory: %s", package, package_dir)
        return None
    return package_dir


def _coverage_data_files(workspace: Path) -> list[Path]:
    """Return coverage data files produced by pytest-cov / coverage.py."""
    return sorted(
        path
        for path in workspace.iterdir()
        if path.is_file() and (path.name == ".coverage" or path.name.startswith(".coverage."))
    )


def _run_logged_command(
    cmd: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run one command, store stdout/stderr in a log file, and return its exit code."""
    display = " ".join(shlex.quote(part) for part in cmd)
    LOGGER.info("$ %s", display)
    completed = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    sections = [f"$ {display}", f"exit_code={completed.returncode}"]
    if completed.stdout:
        sections.extend(["", "stdout:", completed.stdout.rstrip()])
    if completed.stderr:
        sections.extend(["", "stderr:", completed.stderr.rstrip()])
    log_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return completed.returncode


def _normalize_python_xml_filename(raw_path: str, *, repo_source_dir: Path, installed_dirs: tuple[Path, ...]) -> str:
    """Map coverage.py XML filenames back to repo-relative Python source paths."""
    raw = raw_path.strip()
    if not raw:
        return raw

    path = Path(raw)
    if not path.is_absolute():
        stripped = raw.removeprefix("./")
        if stripped.startswith("openvino/"):
            stripped = stripped[len("openvino/") :]
        return Path(stripped).as_posix()

    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path

    for root in (repo_source_dir, *installed_dirs):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue

    parts = resolved.parts
    for marker in ("site-packages", "dist-packages"):
        if marker not in parts:
            continue
        idx = parts.index(marker)
        tail = parts[idx + 1 :]
        if tail and tail[0] == "openvino":
            return Path(*tail[1:]).as_posix()

    if "openvino" in parts:
        openvino_indexes = [idx for idx, part in enumerate(parts) if part == "openvino"]
        for idx in reversed(openvino_indexes):
            tail = parts[idx + 1 :]
            if tail:
                return Path(*tail).as_posix()

    return raw


def _rewrite_python_coverage_xml(*, xml_path: Path, workspace: Path, repo_source_dir: Path, installed_dirs: tuple[Path, ...]) -> None:
    """Rewrite Python XML coverage paths to the repo source layout expected by Codecov."""
    if not xml_path.is_file():
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    try:
        source_value = repo_source_dir.relative_to(workspace).as_posix()
    except ValueError:
        source_value = repo_source_dir.as_posix()

    sources = root.find("sources")
    if sources is not None:
        for source in sources.findall("source"):
            source.text = source_value

    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        cls.attrib["filename"] = _normalize_python_xml_filename(
            filename,
            repo_source_dir=repo_source_dir,
            installed_dirs=installed_dirs,
        )

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _build_cpp_command(exe: Path, *, mode: str, args: str) -> list[str]:
    """Build the command line for one configured C++ test binary."""
    cmd = [str(exe)]
    if mode == "raw":
        if args:
            cmd.extend(shlex.split(args))
        return cmd

    if args:
        cmd.append(f"--gtest_filter={args}")
    return cmd


def _run_cpp_test(
    ctx: CoverageContext,
    test: CppTestCase,
    args: str,
    *,
    runtime_ld_library_path: str,
) -> _TestRunResult:
    """Run one configured C++ test binary."""
    exe = ctx.paths.bin_dir / test.binary
    if not exe.exists():
        return _TestRunResult(test.name, "skipped", f"{test.name} (missing binary: {test.binary})")

    if not os.access(exe, os.X_OK):
        try:
            exe.chmod(exe.stat().st_mode | 0o111)
        except OSError:
            pass

    if not os.access(exe, os.X_OK):
        return _TestRunResult(test.name, "skipped", f"{test.name} (binary not executable: {test.binary})")

    env = env_from_assignments(test.extra_env)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if runtime_ld_library_path:
        env["LD_LIBRARY_PATH"] = f"{runtime_ld_library_path}:{existing_ld}" if existing_ld else runtime_ld_library_path

    rc, duration_seconds = _timed_run(
        f"C++ test: {test.name}",
        _build_cpp_command(exe, mode=test.mode, args=args),
        env=env,
    )
    if rc != 0:
        return _TestRunResult(test.name, "failed", f"{test.name} (exit {rc})", duration_seconds)
    return _TestRunResult(test.name, "passed", duration_seconds=duration_seconds)


def _run_cpp_tests_serial(
    ctx: CoverageContext,
    tests: list[tuple[CppTestCase, str]],
    *,
    runtime_ld_library_path: str,
) -> list[_TestRunResult]:
    """Run selected C++ tests one after another."""
    return [
        _run_cpp_test(
            ctx,
            test,
            args,
            runtime_ld_library_path=runtime_ld_library_path,
        )
        for test, args in tests
    ]


def _normalize_js_source_path(raw_path: str, *, workspace: Path) -> Path | None:
    """Normalize a JS LCOV source path to a repo-relative path."""
    raw = raw_path.strip()
    if not raw:
        return None

    candidates: list[Path] = []
    path = Path(raw)
    if path.is_absolute():
        candidates.append(path)
        parts = path.parts[1:] if path.parts and path.parts[0] == path.anchor else path.parts
        for start in range(len(parts)):
            suffix = parts[start:]
            if suffix:
                candidates.append(workspace.joinpath(*suffix))
    else:
        stripped = raw.removeprefix("./")
        if stripped:
            candidates.append(workspace / stripped)
            repo_prefix = f"{workspace.name}/"
            if stripped.startswith(repo_prefix):
                candidates.append(workspace / stripped[len(repo_prefix) :])
            rel_parts = Path(stripped).parts
            for start in range(len(rel_parts)):
                suffix = rel_parts[start:]
                if suffix:
                    candidates.append(workspace.joinpath(*suffix))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_file():
            try:
                return resolved.relative_to(workspace)
            except ValueError:
                continue
    return None


def _copy_js_lcov(*, source: Path, target: Path, workspace: Path, branch_coverage: bool) -> None:
    """Copy JS LCOV output and optionally strip branch records."""
    output_lines: list[str] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if not branch_coverage and line.startswith(("BRDA:", "BRF:", "BRH:")):
            continue
        if line.startswith("SF:"):
            normalized = _normalize_js_source_path(line[3:].strip(), workspace=workspace)
            if normalized is not None:
                line = f"SF:{normalized.as_posix()}"
        output_lines.append(line)
    target.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def run_cpp(ctx: CoverageContext) -> None:
    """Execute configured C++ tests and record execution statistics."""
    config = CONFIG_DIR / "tests_cpp.yml"
    selected_names = _selected_test_names(CPP_SUITE.selection_env)
    tests = _filter_selected_tests(load_cpp_tests(config, ctx.test_profile), selected_names, suite_label=CPP_SUITE.label)

    _remove_gcda(ctx.paths.build_dir)
    _remove_gcda(ctx.paths.build_js_dir)
    shutil.rmtree(ctx.paths.build_dir / "gcov", ignore_errors=True)

    results_by_name: dict[str, _TestRunResult] = {}
    configured_tests = [(test, test.args.replace("__MODEL_PATH__", str(ctx.paths.model_path))) for test in tests]

    if configured_tests:
        runtime_ld_library_path = _runtime_ld_library_path(ctx)
        for result in _run_cpp_tests_serial(ctx, configured_tests, runtime_ld_library_path=runtime_ld_library_path):
            results_by_name[result.name] = result

    results = [results_by_name[test.name] for test in tests if test.name in results_by_name]

    _finalize_suite(
        ctx,
        CPP_SUITE,
        selected_names=selected_names,
        results=results,
        extra_lines=[
            f"Test profile: {ctx.test_profile}",
            f"GPU mode: {'true' if ctx.run_gpu_tests else 'false'}",
            "",
        ],
        no_execution_warning=(
            f"No C++ tests were executed. Check restored binaries under: {ctx.paths.bin_dir}"
            if results
            else None
        ),
        failure_warning="One or more C++ tests failed; continuing to coverage generation.",
    )


def run_python(ctx: CoverageContext) -> None:
    """Execute configured Python tests and export coverage results."""
    config = CONFIG_DIR / "tests_python.yml"
    selected_names = _selected_test_names(PYTHON_SUITE.selection_env)
    tests = _filter_selected_tests(load_python_tests(config, ctx.test_profile), selected_names, suite_label=PYTHON_SUITE.label)

    results: list[_TestRunResult] = []
    if not tests:
        _finalize_suite(
            ctx,
            PYTHON_SUITE,
            selected_names=selected_names,
            results=results,
            no_execution_warning=f"No Python tests are configured for TEST_PROFILE={ctx.test_profile}; skipping Python suite.",
        )
        return

    tests_dir = ctx.paths.install_pkg_dir / "tests"
    py_source_root = ctx.workspace / "src" / "bindings" / "python" / "src"
    py_source_dir = py_source_root / "openvino"
    layer_tests = ctx.workspace / "tests" / "layer_tests"
    py_cov_config = ctx.workspace / ".python_coverage_ci.rc"
    python_coverage_debug_dir = ctx.workspace / ".tmp" / "python-coverage"

    pip_install_path = os.environ.get("PIP_INSTALL_PATH", "").strip()
    installed_openvino_dir = _detect_installed_python_package_dir("openvino")
    if installed_openvino_dir is None and pip_install_path:
        fallback_dir = Path(pip_install_path) / "openvino"
        if fallback_dir.is_dir():
            installed_openvino_dir = fallback_dir

    py_cov_source_dir = installed_openvino_dir if installed_openvino_dir is not None else py_source_dir
    py_cov_source = str(py_cov_source_dir)
    wheel_lib_dir = Path(pip_install_path) / "openvino" / "libs" if pip_install_path else None
    runtime_extra_paths = [tests_dir]
    if wheel_lib_dir is not None:
        runtime_extra_paths.append(wheel_lib_dir)

    os.environ["LD_LIBRARY_PATH"] = (
        f"{_runtime_ld_library_path(ctx, extra_paths=tuple(runtime_extra_paths))}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    ).rstrip(":")
    python_path_entries = [
        str(tests_dir),
        str(tests_dir / "python"),
        os.environ.get("PYTHONPATH", ""),
    ]
    os.environ["PYTHONPATH"] = ":".join(entry for entry in python_path_entries if entry).rstrip(":")
    os.environ["PYTHONSAFEPATH"] = "1"
    os.environ["TESTS_DIR"] = str(tests_dir)
    os.environ["WORKSPACE_LAYER_TESTS_DIR"] = str(layer_tests)
    os.environ["PY_COV_CONFIG"] = str(py_cov_config)

    py_cov_config.write_text(
        _pycov_config(
            branch_coverage=ctx.branch_coverage,
            runtime_source_dir=py_cov_source_dir,
            repo_source_dir=py_source_dir,
            workspace=ctx.workspace,
            installed_dirs=tuple(path for path in (installed_openvino_dir,) if path is not None),
        ),
        encoding="utf-8",
    )
    os.environ["COVERAGE_RCFILE"] = str(py_cov_config)
    run_cmd(["python3", "-m", "coverage", "erase"])

    for test in tests:
        target = _expand(test.target)
        args = _expand(test.args)
        command = _expand(test.command)
        env = env_from_assignments(_expand(test.env))

        if test.kind == "pytest":
            cmd = _python_pytest_command(target, args, py_cov_source=py_cov_source, py_cov_config=py_cov_config)
            rc, duration_seconds = _timed_run(f"Python test: {test.name}", cmd, env=env)
        elif test.kind == "command":
            rc, duration_seconds = _timed_run(
                f"Python test: {test.name}",
                ["bash", "-lc", command],
                env=env,
            )
        else:
            LOGGER.warning("Unknown Python test kind '%s' for '%s', skipping", test.kind, test.name)
            results.append(_TestRunResult(test.name, "skipped", f"{test.name} (unknown kind: {test.kind})"))
            continue

        if rc != 0:
            results.append(_TestRunResult(test.name, "failed", f"{test.name} (exit {rc})", duration_seconds))
        else:
            results.append(_TestRunResult(test.name, "passed", duration_seconds=duration_seconds))

    coverage_data_files = _coverage_data_files(ctx.workspace)
    if not coverage_data_files:
        raise RuntimeError("Python coverage data files were not produced; cannot export python-coverage.xml")

    primary_coverage_file = ctx.workspace / ".coverage"
    if len(coverage_data_files) > 1 or not primary_coverage_file.is_file():
        combine_log = python_coverage_debug_dir / "coverage-combine.log"
        combine_rc = _run_logged_command(
            ["python3", "-m", "coverage", "combine", "--keep"],
            log_path=combine_log,
            cwd=ctx.workspace,
        )
        if combine_rc != 0:
            raise RuntimeError(
                f"Failed to combine Python coverage data. See {combine_log}"
            )

    xml_path = ctx.workspace / "python-coverage.xml"
    xml_log = python_coverage_debug_dir / "coverage-xml.log"
    xml_rc = _run_logged_command(
        ["python3", "-m", "coverage", "xml", "-i", "-o", str(xml_path)],
        log_path=xml_log,
        cwd=ctx.workspace,
    )
    if xml_rc != 0 or not xml_path.is_file() or xml_path.stat().st_size == 0:
        debug_log = python_coverage_debug_dir / "coverage-debug-data.log"
        _run_logged_command(
            ["python3", "-m", "coverage", "debug", "data"],
            log_path=debug_log,
            cwd=ctx.workspace,
        )
        raise RuntimeError(
            "Failed to export python-coverage.xml; "
            f"see {xml_log} and {debug_log}"
        )

    installed_dirs = tuple(path for path in (installed_openvino_dir,) if path is not None)
    _rewrite_python_coverage_xml(
        xml_path=xml_path,
        workspace=ctx.workspace,
        repo_source_dir=py_source_dir,
        installed_dirs=installed_dirs,
    )

    _finalize_suite(
        ctx,
        PYTHON_SUITE,
        selected_names=selected_names,
        results=results,
        failure_warning="One or more Python tests failed; continuing to coverage generation.",
    )


def run_js(ctx: CoverageContext) -> None:
    """Execute configured JS tests and export coverage results."""
    if shutil.which("node") is None or shutil.which("npm") is None:
        raise RuntimeError("Node.js/npm are not available in the coverage runtime environment.")

    config = CONFIG_DIR / "tests_js.yml"
    selected_names = _selected_test_names(JS_SUITE.selection_env)
    tests = _filter_selected_tests(load_js_tests(config, ctx.test_profile), selected_names, suite_label=JS_SUITE.label)

    results: list[_TestRunResult] = []
    if not tests:
        _finalize_suite(
            ctx,
            JS_SUITE,
            selected_names=selected_names,
            results=results,
            no_execution_warning=f"No JS tests are configured for TEST_PROFILE={ctx.test_profile}; skipping JS suite.",
        )
        return

    os.environ["OV_WORKSPACE"] = str(ctx.workspace)
    os.environ["LD_LIBRARY_PATH"] = (
        f"{_runtime_ld_library_path(ctx, extra_paths=(ctx.paths.js_dir / 'bin',))}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    ).rstrip(":")

    for test in tests:
        if test.kind != "command":
            results.append(_TestRunResult(test.name, "skipped", f"{test.name} (unknown kind: {test.kind})"))
            continue

        rc, duration_seconds = _timed_run(
            f"JS test: {test.name}",
            ["bash", "-lc", _expand(test.command)],
            cwd=ctx.paths.js_dir,
        )
        if rc != 0:
            results.append(_TestRunResult(test.name, "failed", f"{test.name} (exit {rc})", duration_seconds))
        else:
            results.append(_TestRunResult(test.name, "passed", duration_seconds=duration_seconds))

    source = ctx.workspace / "js-coverage" / "lcov.info"
    target = ctx.workspace / "js-lcov.info"
    if source.exists():
        _copy_js_lcov(source=source, target=target, workspace=ctx.workspace, branch_coverage=ctx.branch_coverage)

    _finalize_suite(
        ctx,
        JS_SUITE,
        selected_names=selected_names,
        results=results,
        failure_warning="One or more JS tests failed; continuing to coverage generation.",
    )
