# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path


METADATA_FILE = "coverage-artifact-metadata.json"


SUITE_DEFS = {
    "cpp": {
        "label": "C++",
        "artifacts_dir": "cpp",
        "artifact_glob": "coverage-cpp-*",
        "artifact_prefix": "coverage-cpp-",
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
        "artifact_glob": "coverage-python-*",
        "artifact_prefix": "coverage-python-",
        "stats_file": "python-coverage-stats.env",
        "duration_file": "python-test-durations.csv",
        "coverage_file": "python-coverage.xml",
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
        "artifact_glob": "coverage-js-*",
        "artifact_prefix": "coverage-js-",
        "stats_file": "js-coverage-stats.env",
        "duration_file": "js-test-durations.csv",
        "coverage_file": "js-lcov.info",
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
    raw = values.get(key, "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _artifact_lane(artifact_name: str, prefix: str) -> str:
    suffix = artifact_name[len(prefix) :]
    return suffix.split("-shard-", 1)[0]


def _collect_artifacts(*, workspace: Path, suite_key: str) -> list[dict[str, object]]:
    suite_def = SUITE_DEFS[suite_key]
    root = workspace / "artifacts" / suite_def["artifacts_dir"]
    if not root.exists():
        return []

    artifacts: list[dict[str, object]] = []
    seen_dirs: set[Path] = set()

    for metadata_path in sorted(root.rglob(METADATA_FILE)):
        artifact_dir = metadata_path.parent
        metadata = _read_json_file(metadata_path)
        if not metadata:
            continue
        if str(metadata.get("suite", "")).strip() != suite_key:
            continue
        seen_dirs.add(artifact_dir.resolve())
        artifacts.append(
            {
                "artifact_dir": artifact_dir,
                "artifact_name": str(metadata.get("artifact_name", "")).strip() or artifact_dir.name,
                "lane": str(metadata.get("lane", "")).strip(),
            }
        )

    if artifacts:
        return artifacts

    for artifact_dir in sorted(root.glob(suite_def["artifact_glob"])):
        if not artifact_dir.is_dir():
            continue
        resolved = artifact_dir.resolve()
        if resolved in seen_dirs:
            continue
        artifacts.append(
            {
                "artifact_dir": artifact_dir,
                "artifact_name": artifact_dir.name,
                "lane": _artifact_lane(artifact_dir.name, suite_def["artifact_prefix"]),
            }
        )

    return artifacts


def _flag_patterns(suite: str, lane: str) -> list[str]:
    if suite == "cpp":
        return [f"cpp-runtime-cpp-{lane}-shard-*"]
    if suite == "python":
        return [
            f"python-api-frontend-layer-ovc-{lane}-shard-*",
            f"cpp-runtime-python-{lane}-shard-*",
        ]
    if suite == "js":
        return [
            f"nodejs-bindings-unit-e2e-{lane}-shard-*",
            f"cpp-runtime-js-{lane}-shard-*",
        ]
    raise ValueError(f"Unsupported suite: {suite}")


def render_summary(*, workspace: Path, summary_file: Path, selection: str, selected_lanes: str) -> None:
    rows: list[dict[str, object]] = []
    overall = {"total": 0, "executed": 0, "passed": 0, "failed": 0, "skipped": 0, "not_run": 0, "shards": 0}

    for suite_key, suite_def in SUITE_DEFS.items():
        grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {
                "shards": 0,
                "total": 0,
                "executed": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "not_run": 0,
                "report_size": 0,
            }
        )
        for artifact in _collect_artifacts(workspace=workspace, suite_key=suite_key):
            artifact_dir = Path(artifact["artifact_dir"])
            artifact_name = str(artifact["artifact_name"])
            lane = str(artifact["lane"])
            stats = _read_env_file(artifact_dir / suite_def["stats_file"])
            total = _to_int(stats, suite_def["total_key"])
            passed = _to_int(stats, suite_def["passed_key"])
            failed = _to_int(stats, suite_def["failed_key"])
            skipped = _to_int(stats, suite_def["skipped_key"])
            not_run = _to_int(stats, suite_def["not_run_key"])
            executed = (
                _to_int(stats, suite_def["executed_key"])
                if suite_def["executed_key"]
                else max(0, total - skipped - not_run)
            )
            coverage_file = artifact_dir / suite_def["coverage_file"]
            report_size = coverage_file.stat().st_size if coverage_file.is_file() else 0

            group = grouped[(lane, suite_def["label"])]
            group["shards"] += 1
            group["total"] += total
            group["executed"] += executed
            group["passed"] += passed
            group["failed"] += failed
            group["skipped"] += skipped
            group["not_run"] += not_run
            group["report_size"] += report_size

        for (lane, suite_label), group in sorted(grouped.items()):
            rows.append(
                {
                    "lane": lane,
                    "suite": suite_label,
                    "shards": group["shards"],
                    "total": group["total"],
                    "executed": group["executed"],
                    "passed": group["passed"],
                    "failed": group["failed"],
                    "skipped": group["skipped"],
                    "not_run": group["not_run"],
                    "report_size": group["report_size"],
                    "flags": ", ".join(f"`{flag}`" for flag in _flag_patterns(suite_key, lane)),
                }
            )
            overall["shards"] += group["shards"]
            overall["total"] += group["total"]
            overall["executed"] += group["executed"]
            overall["passed"] += group["passed"]
            overall["failed"] += group["failed"]
            overall["skipped"] += group["skipped"]
            overall["not_run"] += group["not_run"]

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
        f"| Shards | {overall['shards']} |",
        "",
        "### By Lane And Suite",
        "| Lane | Suite | Shards | Total | Executed | Passed | Failed | Skipped | Not run | Report Size (bytes) | Codecov Flags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    if rows:
        for row in rows:
            lines.append(
                f"| {row['lane']} | {row['suite']} | {row['shards']} | {row['total']} | {row['executed']} | "
                f"{row['passed']} | {row['failed']} | {row['skipped']} | {row['not_run']} | {row['report_size']} | {row['flags']} |"
            )
    else:
        lines.append("| - | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |")

    lines.extend(
        [
            "",
            "Merged duration artifact: `coverage-test-durations` (`coverage-test-durations-all.csv`)",
            "",
        ]
    )
    summary_file.write_text("\n".join(lines), encoding="utf-8")


def merge_durations(*, workspace: Path, output: Path) -> None:
    rows: list[dict[str, str]] = []

    for suite_key, suite_def in SUITE_DEFS.items():
        for artifact in _collect_artifacts(workspace=workspace, suite_key=suite_key):
            artifact_dir = Path(artifact["artifact_dir"])
            report = artifact_dir / suite_def["duration_file"]
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

    rows.sort(key=lambda row: (-float(row["duration_seconds"] or 0), row["suite"], row["test_name"]))

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["suite", "lane", "artifact", "test_name", "status", "duration_seconds", "duration_minutes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} duration row(s) to {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage CI report helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render-summary", help="Render the aggregated GitHub summary from downloaded artifacts")
    render.add_argument("--workspace", type=Path, required=True)
    render.add_argument("--summary-file", type=Path, required=True)
    render.add_argument("--selection", required=True)
    render.add_argument("--selected-lanes", required=True)

    merge = subparsers.add_parser("merge-durations", help="Merge shard duration CSV files from downloaded artifacts")
    merge.add_argument("--workspace", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
