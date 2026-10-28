# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare a downloaded CPack BOM artifact against a configuration's GOLD manifests."""

import argparse
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


DOCUMENTATION_URL = (
    "https://github.com/openvinotoolkit/openvino/blob/master/"
    "docs/dev/ci/github_actions/bom_tests.md"
)


@dataclass
class Result:
    name: str
    failure: str = ""
    layout_changed: bool = False


class ManifestError(ValueError):
    pass


def component_files(directory: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise ManifestError(f"Unexpected entry in BOM directory: {path}")
        files[path.stem] = path
    if not files:
        raise ManifestError(f"BOM directory contains no component JSON manifests: {directory}")
    return files


def load_manifest(path: Path) -> set[str]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot read BOM manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("component") != path.stem:
        raise ManifestError(f"Invalid component in {path}; expected component '{path.stem}'")
    files = manifest.get("files")
    if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
        raise ManifestError(f"Invalid files in {path}; expected an array of path strings")
    if len(files) != len(set(files)):
        raise ManifestError(f"Duplicate file paths in BOM manifest {path}")
    return set(files)


def compare(artifact_dir: Path, gold_dir: Path, configuration: str) -> list[Result]:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", configuration):
        return [Result("configuration", failure=f"Invalid configuration name: {configuration!r}")]
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        return [Result(
            "artifact",
            failure=f"Required BOM artifact directory is missing: {artifact_dir}. "
            f"Ensure the Build job uploaded artifact 'bom_{configuration}'.",
        )]

    entries = list(artifact_dir.iterdir())
    if len(entries) != 1 or entries[0].is_symlink() or not entries[0].is_dir():
        return [Result(
            "artifact", failure="BOM artifact must contain exactly one platform directory "
            f"with component JSON manifests: {artifact_dir}",
        )]
    try:
        actual = component_files(entries[0])
    except ManifestError as error:
        return [Result("artifact", failure=str(error))]

    baseline = gold_dir / configuration
    if not baseline.exists():
        # Validate generated manifests even when there is no baseline to compare.
        results = []
        for component, path in actual.items():
            try:
                load_manifest(path)
            except ManifestError as error:
                results.append(Result(component, failure=str(error)))
        results.append(Result(
            "gold-configuration",
            failure=f"No GOLD BOM configuration exists at {baseline}; package layout cannot be verified. "
            "Initialize the configuration's GOLD BOMs through a reviewed PR.",
        ))
        return results
    if baseline.is_symlink() or not baseline.is_dir():
        return [Result("gold-configuration", failure=f"Invalid GOLD configuration directory: {baseline}")]
    try:
        expected = component_files(baseline)
    except ManifestError as error:
        return [Result("gold-configuration", failure=str(error))]

    results = []
    for component in sorted(actual.keys() - expected.keys()):
        results.append(Result(
            f"components.{component}",
            failure=f"New packaged component '{component}'. Add {baseline / (component + '.json')} "
            "from the build's BOM artifact to approve this package-layout change.",
            layout_changed=True,
        ))
    for component in sorted(expected.keys() - actual.keys()):
        results.append(Result(
            f"components.{component}",
            failure=f"Removed packaged component '{component}'. Remove {expected[component]} "
            "to approve this package-layout change, or restore the component's packaging rules.",
            layout_changed=True,
        ))
    if results:
        return results

    results.append(Result("components"))
    for component in sorted(actual):
        try:
            actual_files = load_manifest(actual[component])
            expected_files = load_manifest(expected[component])
        except ManifestError as error:
            results.append(Result(component, failure=str(error)))
            continue
        added = sorted(actual_files - expected_files)
        removed = sorted(expected_files - actual_files)
        if not added and not removed:
            results.append(Result(component))
            continue
        details = [f"Component '{component}' differs from its GOLD BOM:"]
        details.extend(f"  deleted {path} on the PR branch" for path in removed)
        details.extend(f"  added {path} on the PR branch" for path in added)
        if added and removed:
            details.append(
                "Renames appear as a deleted path and an added path; path-only BOMs cannot identify renames."
            )
        results.append(Result(
            component,
            failure="\n".join(details)
            + f"\nUpdate {expected[component]} in your PR to explicitly approve the changed file layout.",
            layout_changed=True,
        ))
    return results


def annotation(level: str, message: str, title: str = "") -> None:
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    properties = f" title={title}" if title else ""
    print(f"::{level}{properties}::{escaped}")


def report_results(results: list[Result], configuration: str, report: Path) -> int:
    failures = sum(bool(result.failure) for result in results)
    suite = ET.Element(
        "testsuite", name=f"BOM {configuration}", tests=str(len(results)),
        failures=str(failures), errors="0", skipped="0",
    )
    for result in results:
        case = ET.SubElement(suite, "testcase", classname=f"bom.{configuration}", name=result.name)
        if result.failure:
            message = f"{configuration}/{result.name}: {result.failure}\nDocumentation: {DOCUMENTATION_URL}"
            title = "BOM package layout changed" if result.layout_changed else "BOM validation failed"
            print(message)
            annotation("error", message, title=title)
            ET.SubElement(case, "failure", message=f"BOM check failed: {result.name}").text = message
        else:
            print(f"PASS: {configuration}/{result.name}")
    report.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(report, encoding="utf-8", xml_declaration=True)
    html_report = report.with_suffix(".html")
    rows = []
    for result in results:
        status = "FAIL" if result.failure else "PASS"
        rows.append(
            f'<tr class="{status.lower()}"><td>{escape(result.name)}</td><td>{status}</td>'
            f"<td><pre>{escape(result.failure)}</pre></td></tr>"
        )
    html_report.write_text(
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>BOM tests: {escape(configuration)}</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #aaa;padding:.5rem;text-align:left;vertical-align:top}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0}"
        ".fail{background:#ffe0e0}.pass{background:#e0ffe0}</style></head><body>"
        f"<h1>BOM tests: {escape(configuration)}</h1>"
        f"<p>{len(results)} checks, {failures} failures.</p>"
        f'<p><a href="{DOCUMENTATION_URL}">BOM documentation and baseline update instructions</a></p>'
        "<table><thead><tr><th>Check</th><th>Status</th><th>Details</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></body></html>\n",
        encoding="utf-8",
    )
    print(f"BOM tests: {len(results)} checks, {failures} failures. Reports: {report}, {html_report}")
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    results = compare(args.artifact_dir, args.gold_dir, args.configuration)
    return report_results(results, args.configuration, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
