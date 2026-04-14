# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "config"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coverage import load_cpp_tests, load_js_tests, load_python_tests


METADATA_FILE = "coverage-artifact-metadata.json"


UPLOAD_DEFS = {
    "cpp_cpu": ("coverage-cpp-cpu", "coverage.info"),
    "cpp_gpu_unit": ("coverage-cpp-gpu-unit", "coverage.info"),
    "cpp_gpu_func": ("coverage-cpp-gpu-func", "coverage.info"),
    "python_cpu_xml": ("coverage-python-cpu", "python-coverage.xml"),
    "python_cpu_info": ("coverage-python-cpu", "coverage.info"),
    "python_gpu_xml": ("coverage-python-gpu", "python-coverage.xml"),
    "python_gpu_info": ("coverage-python-gpu", "coverage.info"),
    "js_cpu_lcov": ("coverage-js-cpu", "js-lcov.info"),
    "js_cpu_info": ("coverage-js-cpu", "coverage.info"),
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
    if suite == "cpp":
        return [test.name for test in load_cpp_tests(CONFIG_DIR / "tests_cpp.yml", profile)]
    if suite == "python":
        return [test.name for test in load_python_tests(CONFIG_DIR / "tests_python.yml", profile)]
    if suite == "js":
        return [test.name for test in load_js_tests(CONFIG_DIR / "tests_js.yml", profile)]
    raise ValueError(f"Unsupported suite: {suite}")


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
    if suite == "cpp":
        return f"`cpp-runtime-cpp-{lane}`"
    if suite == "python":
        return f"`python-api-frontend-layer-ovc-{lane}`, `cpp-runtime-python-{lane}`"
    if suite == "js":
        return f"`nodejs-bindings-unit-e2e-{lane}`, `cpp-runtime-js-{lane}`"
    raise ValueError(f"Unsupported suite: {suite}")


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


def _find_upload_file(*, workspace: Path, artifact_name: str, filename: str) -> Path | None:
    root = workspace / "artifacts"
    if not root.exists():
        return None

    for metadata_path in sorted(root.rglob(METADATA_FILE)):
        metadata = _read_json_file(metadata_path)
        if str(metadata.get("artifact_name", "")).strip() != artifact_name:
            continue
        candidate = metadata_path.parent / filename
        if candidate.is_file():
            return candidate.resolve()

    for candidate in sorted(root.rglob(filename)):
        if candidate.is_file() and artifact_name in candidate.parts:
            return candidate.resolve()

    return None


def resolve_uploads(*, workspace: Path, output_file: Path) -> None:
    output_lines = []
    for output_name, (artifact_name, filename) in UPLOAD_DEFS.items():
        path = _find_upload_file(workspace=workspace, artifact_name=artifact_name, filename=filename)
        output_lines.append(f"{output_name}={path or ''}")
        print(f"{output_name}: {path or '<missing>'}")

    with output_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage CI report helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect-suite-results", help="Prepare one suite artifact with fallback files and real outputs")
    collect.add_argument("--workspace", type=Path, required=True)
    collect.add_argument("--suite", choices=sorted(SUITE_DEFS), required=True)
    collect.add_argument("--profile", required=True)
    collect.add_argument("--lane", required=True)
    collect.add_argument("--artifact-name", required=True)
    collect.add_argument("--artifact-dir", type=Path, required=True)

    render = subparsers.add_parser("render-summary", help="Render the aggregated GitHub summary from downloaded artifacts")
    render.add_argument("--workspace", type=Path, required=True)
    render.add_argument("--summary-file", type=Path, required=True)
    render.add_argument("--selection", required=True)
    render.add_argument("--selected-lanes", required=True)

    merge = subparsers.add_parser("merge-durations", help="Merge suite duration CSV files from downloaded artifacts")
    merge.add_argument("--workspace", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    uploads = subparsers.add_parser("resolve-uploads", help="Resolve Codecov upload files from downloaded artifacts")
    uploads.add_argument("--workspace", type=Path, required=True)
    uploads.add_argument("--output-file", type=Path, required=True)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
