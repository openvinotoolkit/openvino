# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check selected GTest IDs and fixture prerequisites before/after a validation batch."""

import argparse
from collections import Counter
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path


def listed_ids(text):
    suite = None
    result = []
    for line in text.splitlines():
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        if not line[0].isspace():
            suite = value if value.endswith(".") and " " not in value else None
        elif suite and " " not in value:
            result.append(suite + value)
    return result


def aliases(first, second):
    try:
        if first.resolve() == second.resolve():
            return True
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot safely resolve path {first}: {error}") from error
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot safely compare paths {first} and {second}: {error}") from error


def check(contract, listing=None, summary=None):
    selections = contract.get("selections")
    if not isinstance(selections, list) or not selections:
        raise ValueError("contract requires nonempty selections")
    for selection in selections:
        if (not isinstance(selection, dict) or not isinstance(selection.get("pattern"), str)
                or not selection["pattern"] or type(selection.get("count")) is not int
                or selection["count"] < 1):
            raise ValueError("each selection requires a pattern and positive integer count")
    for key in ("allowed_skips", "required_files"):
        values = contract.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(key + " must contain nonempty strings")
    violations = []
    ids = listed_ids(listing) if listing is not None else None
    if ids is not None:
        if not ids:
            violations.append("listing selected zero tests")
        for name, count in Counter(ids).items():
            if count != 1:
                violations.append("duplicate listed ID: " + name)
    executed, skipped = [], []
    if summary is not None:
        if type(summary.get("process_exit")) is not int or summary["process_exit"] != 0:
            violations.append("test process exit is missing or nonzero")
        if summary.get("parse_errors"):
            violations.append("test summary has XML parse errors")
        reports = summary.get("reports")
        if not isinstance(reports, list) or not reports:
            raise ValueError("summary requires nonempty reports")
        for report in reports:
            names = report["selected_ids"]
            if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
                raise ValueError("selected_ids must contain nonempty strings")
            if report["counts"].get("selected") != len(names):
                raise ValueError("selected count and IDs disagree")
            if report.get("suite_errors"):
                violations.append("test report has suite errors")
            executed.extend(names)
            skipped.extend(item["id"] for item in report["skips"])
            if report["counts"].get("failed", 0) or report["counts"].get("errors", 0):
                violations.append("test report has failed cases")
        if not executed:
            violations.append("test reports selected zero tests")
        for name, count in Counter(executed).items():
            if count != 1:
                violations.append("duplicate executed ID: " + name)
        if ids is not None and Counter(ids) != Counter(executed):
            violations.append("listed and executed IDs differ")
        for name in skipped:
            if name not in executed:
                raise ValueError("skipped ID absent from selected IDs")
            if not any(fnmatchcase(name, pattern) for pattern in contract.get("allowed_skips", [])):
                violations.append("unexpected skip: " + name)
    stages = {"listed": ids, "executed": executed if summary is not None else None}
    counts = {}
    for stage, names in stages.items():
        if names is None:
            continue
        counts[stage] = len(names)
        for selection in selections:
            count = sum(fnmatchcase(name, selection["pattern"]) for name in names)
            if count != selection["count"]:
                violations.append(f"{stage} {selection['pattern']}: expected {selection['count']}, got {count}")
    missing = [name for name in contract.get("required_files", []) if not Path(name).is_file()]
    violations.extend("missing required file: " + name for name in missing)
    return {"counts": counts, "listed_ids": ids, "executed_ids": executed,
            "skipped_ids": skipped, "violations": violations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--listing", type=Path, help="Output of the actual filtered binary --gtest_list_tests")
    parser.add_argument("--summary", type=Path, help="Full summarize_tests.py JSON from this attempt")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.listing is None and args.summary is None:
        parser.error("provide --listing and/or --summary")
    inputs = [path for path in (args.contract, args.listing, args.summary) if path is not None]

    def refuse_output(reason):
        print(json.dumps({"counts": {}, "violations": [reason], "omitted_violations": 0,
                          "details": None, "output_refused": True, "reason": reason}))
        return 2

    try:
        if any(aliases(args.output, path) for path in inputs):
            return refuse_output("output must not overwrite an input")
    except ValueError as error:
        return refuse_output(str(error))
    try:
        contract = json.loads(args.contract.read_text())
    except (OSError, ValueError) as error:
        return refuse_output("cannot read or parse contract: " + str(error))
    if not isinstance(contract, dict):
        return refuse_output("contract must be an object")
    required_files = contract.get("required_files", [])
    if (not isinstance(required_files, list)
            or any(not isinstance(name, str) or not name for name in required_files)):
        return refuse_output("required_files must contain nonempty strings")
    fixture_paths = [Path(name) for name in required_files]
    try:
        for path in fixture_paths:
            if aliases(path, args.output):
                return refuse_output("output must not overwrite a required fixture")
    except ValueError as error:
        return refuse_output(str(error))
    if any(not path.is_absolute() for path in fixture_paths):
        return refuse_output("required_files must use absolute paths")
    try:
        result = check(contract, args.listing.read_text() if args.listing else None,
                       json.loads(args.summary.read_text()) if args.summary else None)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = {"counts": {}, "violations": [str(error)], "input_error": True}
    result["inputs"] = {"path": str(args.contract.resolve()),
                       "listing": str(args.listing.resolve()) if args.listing else None,
                       "summary": str(args.summary.resolve()) if args.summary else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": result["counts"], "violations": result["violations"][:5],
                      "omitted_violations": max(0, len(result["violations"]) - 5),
                      "details": str(args.output.resolve())}))
    return 2 if result.get("input_error") else 1 if result["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
