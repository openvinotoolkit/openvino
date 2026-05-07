# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coverage import get_config_dir, resolve_workspace_path


METADATA_FILE = "coverage-artifact-metadata.json"
REQUIRED_REPORTING_FIELDS = (
    "label",
    "stats_file",
    "duration_file",
    "coverage_file",
)
DEFAULT_COUNTER_KEYS = {
    "cpp": {
        "total": "CXX_TESTS_TOTAL",
        "executed": "CXX_TESTS_EXECUTED",
        "passed": "CXX_TESTS_PASSED",
        "failed": "CXX_TESTS_FAILED",
        "skipped": "CXX_TESTS_SKIPPED",
        "not_run": "CXX_TESTS_NOT_RUN",
    },
    "python": {
        "total": "PY_TESTS_TOTAL",
        "executed": None,
        "passed": "PY_TESTS_PASSED",
        "failed": "PY_TESTS_FAILED",
        "skipped": "PY_TESTS_SKIPPED",
        "not_run": "PY_TESTS_NOT_RUN",
    },
    "js": {
        "total": "JS_TESTS_TOTAL",
        "executed": None,
        "passed": "JS_TESTS_PASSED",
        "failed": "JS_TESTS_FAILED",
        "skipped": "JS_TESTS_SKIPPED",
        "not_run": "JS_TESTS_NOT_RUN",
    },
}


def _apply_common_env(args: argparse.Namespace) -> None:
    config_dir = getattr(args, "config_dir", None)
    if config_dir:
        os.environ["COVERAGE_CONFIG_DIR"] = str(resolve_workspace_path(config_dir, workspace=args.workspace.resolve()))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read coverage reporting configs") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def _as_list(value: Any, *, field_name: str, suite: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Suite {suite!r} reporting field {field_name!r} must be a list")
    return value


def _normalize_debug_dirs(value: Any, *, suite: str) -> list[dict[str, str]]:
    debug_dirs: list[dict[str, str]] = []
    for item in _as_list(value, field_name="debug_dirs", suite=suite):
        if not isinstance(item, dict):
            raise ValueError(f"Suite {suite!r} has an invalid debug_dirs entry")
        src = str(item.get("src", "")).strip()
        dest = str(item.get("dest", "")).strip()
        if not src or not dest:
            raise ValueError(f"Suite {suite!r} debug_dirs entries require src and dest")
        debug_dirs.append({"src": src, "dest": dest})
    return debug_dirs


def _normalize_uploads(value: Any, *, suite: str) -> list[dict[str, Any]]:
    uploads: list[dict[str, Any]] = []
    for item in _as_list(value, field_name="uploads", suite=suite):
        if not isinstance(item, dict):
            raise ValueError(f"Suite {suite!r} has an invalid uploads entry")
        upload = dict(item)
        if not str(upload.get("file", "")).strip():
            raise ValueError(f"Suite {suite!r} uploads entries require file")
        if not str(upload.get("flag", upload.get("flag_template", ""))).strip():
            raise ValueError(f"Suite {suite!r} uploads entries require flag or flag_template")
        uploads.append(upload)
    return uploads


def _normalize_suite_def(suite: str, reporting: dict[str, Any], *, config_file: Path | None = None) -> dict[str, Any]:
    missing = [field for field in REQUIRED_REPORTING_FIELDS if not str(reporting.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Suite {suite!r} reporting is missing required field(s): {', '.join(missing)}")

    counter_keys = DEFAULT_COUNTER_KEYS.get(suite)
    if counter_keys is None:
        raise ValueError(f"Suite {suite!r} does not have default reporting counter keys")

    suite_def: dict[str, Any] = {
        "suite": suite,
        "label": str(reporting["label"]).strip(),
        "artifact_group": str(reporting.get("artifact_group", reporting.get("artifacts_dir", suite))).strip() or suite,
        "artifact_name_template": str(reporting.get("artifact_name_template", "coverage-{suite}-{lane}")).strip()
        or "coverage-{suite}-{lane}",
        "stats_file": str(reporting["stats_file"]).strip(),
        "duration_file": str(reporting["duration_file"]).strip(),
        "coverage_file": str(reporting["coverage_file"]).strip(),
        "selection_env": str(reporting.get("selection_env", "")).strip(),
        "extra_files": [str(item) for item in _as_list(reporting.get("extra_files"), field_name="extra_files", suite=suite)],
        "debug_dirs": _normalize_debug_dirs(reporting.get("debug_dirs"), suite=suite),
        "uploads": _normalize_uploads(reporting.get("uploads"), suite=suite),
        "total_key": str(counter_keys["total"]),
        "executed_key": str(counter_keys["executed"]) if counter_keys["executed"] else None,
        "passed_key": str(counter_keys["passed"]),
        "failed_key": str(counter_keys["failed"]),
        "skipped_key": str(counter_keys["skipped"]),
        "not_run_key": str(counter_keys["not_run"]),
    }

    tests_file = str(reporting.get("tests_file", "")).strip()
    if tests_file:
        suite_def["tests_file"] = tests_file
    elif config_file is not None:
        suite_def["tests_file"] = str(config_file)
    return suite_def


def _load_suite_defs() -> dict[str, dict[str, Any]]:
    suite_defs: dict[str, dict[str, Any]] = {}
    config_dir = get_config_dir()
    for path in sorted(config_dir.glob("tests_*.yml")):
        data = _load_yaml(path)
        suite = str(data.get("suite", "")).strip()
        if not suite:
            raise ValueError(f"Coverage config {path} is missing suite")
        reporting = data.get("reporting")
        if not isinstance(reporting, dict):
            raise ValueError(f"Coverage config {path} is missing reporting section")
        suite_defs[suite] = _normalize_suite_def(suite, reporting, config_file=path)
    if not suite_defs:
        raise ValueError(f"No coverage reporting suite definitions were found in {config_dir}")
    return suite_defs


def _get_suite_def(suite: str) -> dict[str, Any]:
    suite_defs = _load_suite_defs()
    if suite not in suite_defs:
        available = ", ".join(sorted(suite_defs))
        raise ValueError(f"Suite {suite!r} is not defined in coverage reporting configs. Available suites: {available}")
    return suite_defs[suite]


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


def _load_tests_file_names(path: Path, profile: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Suite tests file does not exist: {path}")

    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read suite tests files") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tests = data.get("tests", [])
    if not isinstance(tests, list):
        raise ValueError(f"Suite tests file {path} must contain a 'tests' list")

    names: list[str] = []
    for item in tests:
        if not isinstance(item, dict):
            continue
        profiles = item.get("profiles", [])
        if profiles and profile not in {str(profile_item) for profile_item in profiles}:
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _load_test_names(*, workspace: Path, suite_def: dict[str, Any], profile: str) -> list[str]:
    test_names = suite_def.get("test_names")
    if isinstance(test_names, list):
        return [str(name).strip() for name in test_names if str(name).strip()]

    tests_file = str(suite_def.get("tests_file", "")).strip()
    if tests_file:
        return _load_tests_file_names(resolve_workspace_path(tests_file, workspace=workspace), profile)

    return []


def _selected_test_names(*, suite_def: dict[str, Any], test_names: str) -> list[str]:
    raw_value = test_names.strip()
    if not raw_value:
        selection_env = str(suite_def.get("selection_env", "")).strip()
        raw_value = os.environ.get(selection_env, "").strip() if selection_env else ""
    if not raw_value:
        return []
    return [name.strip() for name in raw_value.split(",") if name.strip()]


def collect_suite_results(
    *,
    workspace: Path,
    suite: str,
    profile: str,
    lane: str,
    test_names: str,
    artifact_name: str,
    artifact_dir: Path | None,
    outputs_file: Path | None,
) -> tuple[str, Path]:
    suite_def = _get_suite_def(suite)
    lane = lane or profile
    artifact_name = artifact_name or _format_template(
        str(suite_def["artifact_name_template"]),
        _template_context(suite_def=suite_def, profile=profile, lane=lane),
    )
    if artifact_dir is None:
        runner_temp = os.environ.get("RUNNER_TEMP", "").strip()
        artifact_dir = Path(runner_temp or workspace) / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tests = _load_test_names(workspace=workspace, suite_def=suite_def, profile=profile)
    selected_tests = _selected_test_names(suite_def=suite_def, test_names=test_names)
    if selected_tests:
        selected_set = set(selected_tests)
        tests = [test for test in tests if test in selected_set]
    total = len(tests)

    metadata = {
        "suite": suite,
        "profile": profile,
        "lane": lane,
        "artifact_name": artifact_name,
        "artifact_group": suite_def["artifact_group"],
    }
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

    for debug_dir in suite_def.get("debug_dirs", []):
        source = resolve_workspace_path(str(debug_dir["src"]), workspace=workspace)
        if source.is_dir():
            destination = artifact_dir / str(debug_dir["dest"])
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination)

    if outputs_file is not None:
        with outputs_file.open("a", encoding="utf-8") as handle:
            handle.write(f"dir={artifact_dir}\n")
            handle.write(f"artifact-dir={artifact_dir}\n")
            handle.write(f"artifact-name={artifact_name}\n")

    return artifact_name, artifact_dir


def _collect_artifacts(*, workspace: Path, suite_key: str) -> list[dict[str, object]]:
    suite_def = _get_suite_def(suite_key)
    root = workspace / "artifacts" / str(suite_def["artifact_group"])
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
                "artifact_group": str(metadata.get("artifact_group", suite_def["artifact_group"])).strip(),
                "lane": str(metadata.get("lane", "")).strip() or "-",
                "profile": str(metadata.get("profile", "")).strip(),
                "suite": suite_key,
            }
        )
    return artifacts


def _template_context(
    *,
    suite_def: dict[str, Any],
    profile: str = "",
    lane: str = "",
    artifact_name: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    context = {
        "suite": str(suite_def["suite"]),
        "label": str(suite_def["label"]),
        "artifact_group": str(suite_def["artifact_group"]),
        "profile": profile,
        "lane": lane,
        "artifact_name": artifact_name,
    }
    if extra:
        context.update({str(key): str(value) for key, value in extra.items()})
    return context


def _artifact_context(suite_def: dict[str, Any], artifact: dict[str, object]) -> dict[str, str]:
    return _template_context(
        suite_def=suite_def,
        profile=str(artifact.get("profile", "")),
        lane=str(artifact.get("lane", "")),
        artifact_name=str(artifact.get("artifact_name", "")),
    )


def _format_template(template: str, context: dict[str, str]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        available = ", ".join(sorted(context))
        raise ValueError(f"Unknown template variable {exc!s}; available variables: {available}") from exc


def _format_upload_value(upload: dict[str, Any], *, field: str, template_field: str, context: dict[str, str], default: str = "") -> str:
    template = str(upload.get(template_field, upload.get(field, default))).strip()
    return _format_template(template, context) if template else ""


def _artifact_upload_flags(suite_def: dict[str, Any], artifact: dict[str, object]) -> str:
    flags: list[str] = []
    artifact_name = str(artifact.get("artifact_name", ""))
    context = _artifact_context(suite_def, artifact)
    for upload in suite_def.get("uploads", []):
        pattern = str(upload.get("artifact_pattern", "")).strip()
        if pattern and not _artifact_name_matches(artifact_name, _format_template(pattern, context)):
            continue
        flag = _format_upload_value(upload, field="flag", template_field="flag_template", context=context)
        if flag and flag not in flags:
            flags.append(flag)
    return ", ".join(f"`{flag}`" for flag in flags) if flags else "-"


def render_summary(*, workspace: Path, summary_file: Path, selection: str, selected_lanes: str) -> None:
    rows: list[dict[str, object]] = []
    overall = {"total": 0, "executed": 0, "passed": 0, "failed": 0, "skipped": 0, "not_run": 0}
    suite_defs = _load_suite_defs()

    for suite_key, suite_def in suite_defs.items():
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
                    "flags": _artifact_upload_flags(suite_def, artifact),
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
    suite_defs = _load_suite_defs()
    for suite_key, suite_def in suite_defs.items():
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
    return fnmatchcase(artifact_name, pattern)


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _upload_key(upload: dict[str, Any], context: dict[str, str], *, file_name: str) -> str:
    context = {**context, "file": file_name}
    raw_key = _format_upload_value(
        upload,
        field="key",
        template_field="key_template",
        context=context,
        default="{suite}_{lane}_{file}",
    )
    return "".join(character if character.isalnum() else "_" for character in raw_key).strip("_")


def _upload_entry(
    *,
    suite_def: dict[str, Any],
    upload: dict[str, Any],
    artifacts: list[dict[str, object]],
    workspace: Path,
    context: dict[str, str],
    artifact_pattern: str,
) -> dict[str, Any] | None:
    file_name = _format_upload_value(upload, field="file", template_field="file_template", context=context)
    if not file_name:
        raise ValueError(f"Upload entry for suite {suite_def['suite']!r} resolved to an empty file name")

    files: list[str] = []
    artifact_names: list[str] = []
    for artifact in artifacts:
        artifact_name = str(artifact["artifact_name"])
        if not _artifact_name_matches(artifact_name, artifact_pattern):
            continue
        candidate = Path(artifact["artifact_dir"]) / file_name
        if candidate.is_file():
            files.append(_workspace_relative(candidate, workspace))
            artifact_names.append(artifact_name)

    if not files:
        return None

    flag = _format_upload_value(upload, field="flag", template_field="flag_template", context=context)
    name = _format_upload_value(
        upload,
        field="name",
        template_field="name_template",
        context=context,
        default="{label}, {lane}",
    )

    return {
        "key": _upload_key(upload, context, file_name=file_name),
        "name": name,
        "suite": suite_def["suite"],
        "artifact_group": suite_def["artifact_group"],
        "artifact_pattern": artifact_pattern,
        "artifact_names": artifact_names,
        "file": file_name,
        "files": files,
        "files_csv": ",".join(files),
        "flag": flag,
    }


def _build_upload_matrix(*, workspace: Path) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    suite_defs = _load_suite_defs()
    for suite_key, suite_def in suite_defs.items():
        artifacts = _collect_artifacts(workspace=workspace, suite_key=suite_key)
        if not artifacts:
            continue

        for upload in suite_def.get("uploads", []):
            artifact_pattern_template = str(upload.get("artifact_pattern", "")).strip()
            if artifact_pattern_template:
                context = _template_context(
                    suite_def=suite_def,
                    profile=str(upload.get("profile", "")),
                    lane=str(upload.get("lane", "")),
                    artifact_name=str(upload.get("artifact_name", "")),
                    extra={key: value for key, value in upload.items() if isinstance(value, (str, int, float))},
                )
                artifact_pattern = _format_template(artifact_pattern_template, context)
                entry = _upload_entry(
                    suite_def=suite_def,
                    upload=upload,
                    artifacts=artifacts,
                    workspace=workspace,
                    context=context,
                    artifact_pattern=artifact_pattern,
                )
                if entry:
                    matrix.append(entry)
                continue

            for artifact in artifacts:
                context = _artifact_context(suite_def, artifact)
                artifact_pattern = str(artifact["artifact_name"])
                entry = _upload_entry(
                    suite_def=suite_def,
                    upload=upload,
                    artifacts=[artifact],
                    workspace=workspace,
                    context=context,
                    artifact_pattern=artifact_pattern,
                )
                if entry:
                    matrix.append(entry)
    return matrix


def resolve_uploads(*, workspace: Path, output_file: Path) -> None:
    upload_matrix = _build_upload_matrix(workspace=workspace)
    upload_files = {str(entry["key"]): ",".join(entry["files"]) for entry in upload_matrix}
    output_lines = [
        f"upload-files-json={json.dumps(upload_files, sort_keys=True)}",
        f"upload-matrix-json={json.dumps(upload_matrix, sort_keys=True)}",
    ]

    for entry in upload_matrix:
        print(f"{entry['name']}: {','.join(entry['files'])}")
    if not upload_matrix:
        print("No Codecov upload files were resolved")

    with output_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage CI report helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config-dir", default=os.environ.get("COVERAGE_CONFIG_DIR"))

    collect = subparsers.add_parser(
        "collect-suite-results",
        parents=[common],
        help="Prepare one suite artifact with fallback files and real outputs",
    )
    collect.add_argument("--workspace", type=Path, required=True)
    collect.add_argument("--suite", required=True)
    collect.add_argument("--profile", required=True)
    collect.add_argument("--lane", required=True)
    collect.add_argument("--test-names", default="")
    collect.add_argument("--artifact-name", default="")
    collect.add_argument("--artifact-dir", type=Path)
    collect.add_argument("--outputs-file", type=Path)

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
            test_names=args.test_names,
            artifact_name=args.artifact_name,
            artifact_dir=args.artifact_dir.resolve() if args.artifact_dir else None,
            outputs_file=args.outputs_file.resolve() if args.outputs_file else None,
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
