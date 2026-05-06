# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coverage import get_config_dir, load_cpp_tests, load_js_tests, load_python_tests, resolve_workspace_path


METADATA_FILE = "coverage-artifact-metadata.json"


UPLOAD_DEFS = {
    "cpp_cpu": ("coverage-cpp-cpu", "coverage.info"),
    "cpp_igpu_unit": ("coverage-cpp-igpu-unit", "coverage.info"),
    "cpp_igpu_func": ("coverage-cpp-igpu-func", "coverage.info"),
    "cpp_dgpu_unit": ("coverage-cpp-dgpu-unit", "coverage.info"),
    "cpp_dgpu_func": ("coverage-cpp-dgpu-func", "coverage.info"),
    "python_cpu_xml": ("coverage-python-cpu", "python-coverage.xml"),
    "python_cpu_info": ("coverage-python-cpu", "coverage.info"),
    "python_igpu_xml": ("coverage-python-igpu", "python-coverage.xml"),
    "python_igpu_info": ("coverage-python-igpu", "coverage.info"),
    "python_dgpu_xml": ("coverage-python-dgpu", "python-coverage.xml"),
    "python_dgpu_info": ("coverage-python-dgpu", "coverage.info"),
    "js_cpu_lcov": ("coverage-js-cpu", "js-lcov.info"),
    "js_cpu_info": ("coverage-js-cpu", "coverage.info"),
}


FLAG_PATTERNS = {
    "cpp": ["cpp-runtime-cpp-{lane}"],
    "python": ["python-api-frontend-layer-ovc-{lane}", "cpp-runtime-python-{lane}"],
    "js": ["nodejs-bindings-unit-e2e-{lane}", "cpp-runtime-js-{lane}"],
}


SUITE_DEFS = {
    "cpp": {
        "label": "C++",
        "artifacts_dir": "cpp",
        "stats_file": "cpp-coverage-stats.env",
        "duration_file": "cpp-test-durations.csv",
        "coverage_file": "coverage.info",
        "total_key": "CXX_TESTS_TOTAL",
        "executed_key": "CXX_TESTS_EXECUTED",
        "passed_key": "CXX_TESTS_PASSED",
        "failed_key": "CXX_TESTS_FAILED",
        "skipped_key": "CXX_TESTS_SKIPPED",
        "not_run_key": "CXX_TESTS_NOT_RUN",
    },
    "python": {
        "label": "Python",
        "artifacts_dir": "python",
        "stats_file": "python-coverage-stats.env",
        "duration_file": "python-test-durations.csv",
        "coverage_file": "python-coverage.xml",
        "extra_files": ["coverage.info"],
        "total_key": "PY_TESTS_TOTAL",
        "executed_key": None,
        "passed_key": "PY_TESTS_PASSED",
        "failed_key": "PY_TESTS_FAILED",
        "skipped_key": "PY_TESTS_SKIPPED",
        "not_run_key": "PY_TESTS_NOT_RUN",
    },
    "js": {
        "label": "JS",
        "artifacts_dir": "js",
        "stats_file": "js-coverage-stats.env",
        "duration_file": "js-test-durations.csv",
        "coverage_file": "js-lcov.info",
        "extra_files": ["coverage.info"],
        "total_key": "JS_TESTS_TOTAL",
        "executed_key": None,
        "passed_key": "JS_TESTS_PASSED",
        "failed_key": "JS_TESTS_FAILED",
        "skipped_key": "JS_TESTS_SKIPPED",
        "not_run_key": "JS_TESTS_NOT_RUN",
    },
}


SELECTION_ENV_BY_SUITE = {
    "cpp": "CXX_TEST_NAMES",
    "python": "PY_TEST_NAMES",
    "js": "JS_TEST_NAMES",
}


def _apply_common_env(args: argparse.Namespace) -> None:
    config_dir = getattr(args, "config_dir", None)
    if config_dir:
        os.environ["COVERAGE_CONFIG_DIR"] = str(resolve_workspace_path(config_dir, workspace=args.workspace.resolve()))

    upload_defs_json = getattr(args, "upload_defs_json", None)
    if upload_defs_json:
        os.environ["COVERAGE_UPLOAD_DEFS_JSON"] = upload_defs_json

    flag_patterns_json = getattr(args, "flag_patterns_json", None)
    if flag_patterns_json:
        os.environ["COVERAGE_FLAG_PATTERNS_JSON"] = flag_patterns_json


def _load_upload_defs() -> dict[str, tuple[str, str]]:
    raw = os.environ.get("COVERAGE_UPLOAD_DEFS_JSON", "").strip()
    if not raw:
        return UPLOAD_DEFS

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("COVERAGE_UPLOAD_DEFS_JSON must be valid JSON") from exc

    if not isinstance(loaded, dict):
        raise ValueError("COVERAGE_UPLOAD_DEFS_JSON must be a JSON object")

    upload_defs: dict[str, tuple[str, str]] = {}
    for output_name, value in loaded.items():
        if isinstance(value, dict):
            artifact = str(value.get("artifact", "")).strip()
            filename = str(value.get("file", "")).strip()
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            artifact = str(value[0]).strip()
            filename = str(value[1]).strip()
        else:
            raise ValueError(f"Invalid upload definition for {output_name!r}")
        if not artifact or not filename:
            raise ValueError(f"Upload definition for {output_name!r} requires artifact and file")
        upload_defs[str(output_name)] = (artifact, filename)

    return upload_defs


def _load_flag_patterns() -> dict[str, list[str]]:
    raw = os.environ.get("COVERAGE_FLAG_PATTERNS_JSON", "").strip()
    if not raw:
        return FLAG_PATTERNS

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("COVERAGE_FLAG_PATTERNS_JSON must be valid JSON") from exc

    if not isinstance(loaded, dict):
        raise ValueError("COVERAGE_FLAG_PATTERNS_JSON must be a JSON object")

    patterns: dict[str, list[str]] = {}
    for suite, value in loaded.items():
        if isinstance(value, str):
            patterns[str(suite)] = [value]
        elif isinstance(value, list):
            patterns[str(suite)] = [str(item) for item in value]
        else:
            raise ValueError(f"Invalid flag pattern definition for {suite!r}")
    return patterns


def _read_json_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _to_int(values: dict[str, str], key: str | None) -> int:
    if not key:
        return 0
    try:
        return int(values.get(key, "0").strip())
    except ValueError:
        return 0


def _load_test_names(*, workspace: Path, suite: str, profile: str) -> list[str]:
    config_dir = get_config_dir()
    if suite == "cpp":
        return [test.name for test in load_cpp_tests(config_dir / "tests_cpp.yml", profile)]
    if suite == "python":
        return [test.name for test in load_python_tests(config_dir / "tests_python.yml", profile)]
    if suite == "js":
        return [test.name for test in load_js_tests(config_dir / "tests_js.yml", profile)]
    raise ValueError(f"Unsupported suite: {suite}")


def _selected_test_names(suite: str) -> list[str]:
    raw_value = os.environ.get(SELECTION_ENV_BY_SUITE[suite], "").strip()
    if not raw_value:
        return []
    return [name.strip() for name in raw_value.split(",") if name.strip()]


def collect_suite_results(
    *,
    workspace: Path,
    suite: str,
    profile: str,
    lane: str,
    artifact_name: str,
    artifact_dir: Path,
) -> None:
    suite_def = SUITE_DEFS[suite]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tests = _load_test_names(workspace=workspace, suite=suite, profile=profile)
    selected_tests = _selected_test_names(suite)
    if selected_tests:
        selected_set = set(selected_tests)
        tests = [test for test in tests if test in selected_set]
    total = len(tests)

    metadata = {"suite": suite, "lane": lane, "artifact_name": artifact_name}
    (artifact_dir / METADATA_FILE).write_text(json.dumps(metadata, separators=(",", ":")) + "\n", encoding="utf-8")

    duration_path = artifact_dir / str(suite_def["duration_file"])
    with duration_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_name", "status", "duration_seconds", "duration_minutes"])
        for test_name in tests:
            writer.writerow([test_name, "not_run", "0.000", "0.000"])

    stats_lines = [f"{suite_def['total_key']}={total}"]
    if suite_def["executed_key"]:
        stats_lines.append(f"{suite_def['executed_key']}=0")
    stats_lines.extend(
        [
            f"{suite_def['passed_key']}=0",
            f"{suite_def['failed_key']}=0",
            f"{suite_def['skipped_key']}=0",
            f"{suite_def['not_run_key']}={total}",
        ]
    )
    (artifact_dir / str(suite_def["stats_file"])).write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    files_to_copy = [
        str(suite_def["duration_file"]),
        str(suite_def["stats_file"]),
        str(suite_def["coverage_file"]),
        *[str(name) for name in suite_def.get("extra_files", [])],
    ]
    for filename in files_to_copy:
        source = workspace / filename
        if source.is_file():
            shutil.copy2(source, artifact_dir / filename)

    debug_dir = workspace / ".tmp" / "cpp-coverage-parts"
    if debug_dir.is_dir():
        destination = artifact_dir / "native-cpp-debug"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(debug_dir, destination)

    python_debug_dir = workspace / ".tmp" / "python-coverage"
    if suite == "python" and python_debug_dir.is_dir():
        destination = artifact_dir / "python-coverage-debug"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(python_debug_dir, destination)


def _collect_artifacts(*, workspace: Path, suite_key: str) -> list[dict[str, object]]:
    suite_def = SUITE_DEFS[suite_key]
    root = workspace / "artifacts" / suite_def["artifacts_dir"]
    if not root.exists():
        return []

    artifacts: list[dict[str, object]] = []
    for metadata_path in sorted(root.rglob(METADATA_FILE)):
        metadata = _read_json_file(metadata_path)
        if not metadata or str(metadata.get("suite", "")).strip() != suite_key:
            continue
        artifacts.append(
            {
                "artifact_dir": metadata_path.parent,
                "artifact_name": str(metadata.get("artifact_name", metadata_path.parent.name)).strip(),
                "lane": str(metadata.get("lane", "")).strip() or "-",
            }
        )
    return artifacts


def _flag_patterns(suite: str, lane: str) -> str:
    patterns = _load_flag_patterns().get(suite)
    if patterns is None:
        raise ValueError(f"Unsupported suite: {suite}")
    return ", ".join(f"`{pattern.format(lane=lane)}`" for pattern in patterns)


def render_summary(*, workspace: Path, summary_file: Path, selection: str, selected_lanes: str) -> None:
    rows: list[dict[str, object]] = []
    overall = {"total": 0, "executed": 0, "passed": 0, "failed": 0, "skipped": 0, "not_run": 0}

    for suite_key, suite_def in SUITE_DEFS.items():
        for artifact in _collect_artifacts(workspace=workspace, suite_key=suite_key):
            artifact_dir = Path(artifact["artifact_dir"])
            lane = str(artifact["lane"])
            stats = _read_env_file(artifact_dir / str(suite_def["stats_file"]))
            total = _to_int(stats, str(suite_def["total_key"]))
            passed = _to_int(stats, str(suite_def["passed_key"]))
            failed = _to_int(stats, str(suite_def["failed_key"]))
            skipped = _to_int(stats, str(suite_def["skipped_key"]))
            not_run = _to_int(stats, str(suite_def["not_run_key"]))
            executed = _to_int(stats, suite_def["executed_key"]) if suite_def["executed_key"] else max(0, total - skipped - not_run)
            coverage_file = artifact_dir / str(suite_def["coverage_file"])
            report_size = coverage_file.stat().st_size if coverage_file.is_file() else 0

            rows.append(
                {
                    "lane": lane,
                    "suite": suite_def["label"],
                    "total": total,
                    "executed": executed,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "not_run": not_run,
                    "report_size": report_size,
                    "flags": _flag_patterns(suite_key, lane),
                }
            )
            overall["total"] += total
            overall["executed"] += executed
            overall["passed"] += passed
            overall["failed"] += failed
            overall["skipped"] += skipped
            overall["not_run"] += not_run

    pass_rate = (overall["passed"] * 100.0) / overall["executed"] if overall["executed"] else 0.0

    lines = [
        "## Coverage Summary",
        "",
        f"**Selection:** `{selection}`",
        "",
        f"**Active lanes:** `{selected_lanes}`",
        "",
        f"**Executed pass rate:** `{pass_rate:.1f}%`",
        "",
        "### Overall",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total test units | {overall['total']} |",
        f"| Executed | {overall['executed']} |",
        f"| Passed | {overall['passed']} |",
        f"| Failed | {overall['failed']} |",
        f"| Skipped | {overall['skipped']} |",
        f"| Not run | {overall['not_run']} |",
        "",
        "### By Lane And Suite",
        "| Lane | Suite | Total | Executed | Passed | Failed | Skipped | Not run | Report Size (bytes) | Codecov Flags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    if rows:
        for row in sorted(rows, key=lambda item: (str(item["lane"]), str(item["suite"]))):
            lines.append(
                f"| {row['lane']} | {row['suite']} | {row['total']} | {row['executed']} | "
                f"{row['passed']} | {row['failed']} | {row['skipped']} | {row['not_run']} | {row['report_size']} | {row['flags']} |"
            )
    else:
        lines.append("| - | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |")

    lines.extend(["", "Merged duration artifact: `coverage-test-durations` (`coverage-test-durations-all.csv`)", ""])
    summary_file.write_text("\n".join(lines), encoding="utf-8")


def merge_durations(*, workspace: Path, output: Path) -> None:
    rows: list[dict[str, str]] = []
    for suite_key, suite_def in SUITE_DEFS.items():
        for artifact in _collect_artifacts(workspace=workspace, suite_key=suite_key):
            artifact_dir = Path(artifact["artifact_dir"])
            report = artifact_dir / str(suite_def["duration_file"])
            if not report.is_file():
                continue
            artifact_name = str(artifact["artifact_name"])
            lane = str(artifact["lane"])
            with report.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(
                        {
                            "suite": suite_key,
                            "lane": lane,
                            "artifact": artifact_name,
                            "test_name": row.get("test_name", ""),
                            "status": row.get("status", ""),
                            "duration_seconds": row.get("duration_seconds", "0"),
                            "duration_minutes": row.get("duration_minutes", "0"),
                        }
                    )

    rows.sort(key=lambda row: (-float(row["duration_seconds"] or 0), row["suite"], row["lane"], row["test_name"]))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["suite", "lane", "artifact", "test_name", "status", "duration_seconds", "duration_minutes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} duration row(s) to {output}")


def _artifact_name_matches(artifact_name: str, pattern: str) -> bool:
    if pattern.endswith("*"):
        return artifact_name.startswith(pattern[:-1])
    return artifact_name == pattern


def _find_upload_files(*, workspace: Path, artifact_name: str, filename: str) -> list[Path]:
    root = workspace / "artifacts"
    if not root.exists():
        return []

    paths: list[Path] = []
    for metadata_path in sorted(root.rglob(METADATA_FILE)):
        metadata = _read_json_file(metadata_path)
        if not _artifact_name_matches(str(metadata.get("artifact_name", "")).strip(), artifact_name):
            continue
        candidate = metadata_path.parent / filename
        if candidate.is_file():
            paths.append(candidate.resolve())

    for candidate in sorted(root.rglob(filename)):
        if not candidate.is_file() or candidate.resolve() in paths:
            continue
        candidate_parts = [str(part) for part in candidate.parts]
        if any(_artifact_name_matches(part, artifact_name) for part in candidate_parts):
            paths.append(candidate.resolve())

    return paths


def resolve_uploads(*, workspace: Path, output_file: Path) -> None:
    output_lines = []
    for output_name, (artifact_name, filename) in _load_upload_defs().items():
        paths = _find_upload_files(workspace=workspace, artifact_name=artifact_name, filename=filename)
        output_value = ",".join(str(path) for path in paths)
        output_lines.append(f"{output_name}={output_value}")
        print(f"{output_name}: {output_value or '<missing>'}")

    with output_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage CI report helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config-dir", default=os.environ.get("COVERAGE_CONFIG_DIR"))
    common.add_argument("--upload-defs-json", default=os.environ.get("COVERAGE_UPLOAD_DEFS_JSON"))
    common.add_argument("--flag-patterns-json", default=os.environ.get("COVERAGE_FLAG_PATTERNS_JSON"))

    collect = subparsers.add_parser(
        "collect-suite-results",
        parents=[common],
        help="Prepare one suite artifact with fallback files and real outputs",
    )
    collect.add_argument("--workspace", type=Path, required=True)
    collect.add_argument("--suite", choices=sorted(SUITE_DEFS), required=True)
    collect.add_argument("--profile", required=True)
    collect.add_argument("--lane", required=True)
    collect.add_argument("--artifact-name", required=True)
    collect.add_argument("--artifact-dir", type=Path, required=True)

    render = subparsers.add_parser(
        "render-summary",
        parents=[common],
        help="Render the aggregated GitHub summary from downloaded artifacts",
    )
    render.add_argument("--workspace", type=Path, required=True)
    render.add_argument("--summary-file", type=Path, required=True)
    render.add_argument("--selection", required=True)
    render.add_argument("--selected-lanes", required=True)

    merge = subparsers.add_parser(
        "merge-durations",
        parents=[common],
        help="Merge suite duration CSV files from downloaded artifacts",
    )
    merge.add_argument("--workspace", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    uploads = subparsers.add_parser(
        "resolve-uploads",
        parents=[common],
        help="Resolve Codecov upload files from downloaded artifacts",
    )
    uploads.add_argument("--workspace", type=Path, required=True)
    uploads.add_argument("--output-file", type=Path, required=True)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _apply_common_env(args)
    if args.command == "collect-suite-results":
        collect_suite_results(
            workspace=args.workspace.resolve(),
            suite=args.suite,
            profile=args.profile,
            lane=args.lane,
            artifact_name=args.artifact_name,
            artifact_dir=args.artifact_dir.resolve(),
        )
        return 0
    if args.command == "render-summary":
        render_summary(
            workspace=args.workspace.resolve(),
            summary_file=args.summary_file.resolve(),
            selection=args.selection,
            selected_lanes=args.selected_lanes,
        )
        return 0
    if args.command == "merge-durations":
        merge_durations(workspace=args.workspace.resolve(), output=args.output.resolve())
        return 0
    if args.command == "resolve-uploads":
        resolve_uploads(workspace=args.workspace.resolve(), output_file=args.output_file.resolve())
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
