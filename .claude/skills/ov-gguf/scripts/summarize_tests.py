# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Retain GTest/JUnit XML details and print a bounded summary; exit codes indicate parsing only."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def signature(message):
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    useful = [line for line in lines if line != "unknown file" and "Exception from " not in line and not re.fullmatch(r".*:\d+(?::\d+)?", line)]
    return " ".join((useful or lines)[:2])[:320] or "No message provided"


def aliases(first, second):
    if first.resolve() == second.resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def summarize(path):
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    cases = list(root.iter("testcase"))
    counts = Counter(selected=0, passed=0, failed=0, errors=0, skipped=0, not_run=0)
    failures, skips, selected_ids = [], [], []
    for case in cases:
        if case.get("status") == "notrun" or case.get("result") == "suppressed":
            counts["not_run"] += 1
            continue
        counts["selected"] += 1
        name = ".".join(filter(None, (case.get("classname"), case.get("name"))))
        selected_ids.append(name)
        errors = case.findall("error")
        failed = case.findall("failure")
        skipped = case.findall("skipped")
        counts["errors" if errors else "failed" if failed else "skipped" if skipped else "passed"] += 1
        for node in errors + failed:
            message = node.text or node.get("message", "")
            failures.append({"id": name, "kind": node.tag, "message": message, "signature": signature(message)})
        for node in skipped:
            skips.append({"id": name, "message": node.text or node.get("message", "")})
    suite_errors = [
        {"kind": node.tag, "message": node.text or node.get("message", "")}
        for suite in root.iter("testsuite")
        for node in list(suite)
        if node.tag in ("failure", "error")
    ]
    return {"path": str(path.resolve()), "counts": dict(counts), "selected_ids": selected_ids, "failures": failures,
            "skips": skips, "suite_errors": suite_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Full JSON details; must differ from input reports")
    parser.add_argument("--process-exit", type=int, help="Recorded exit code of the test process")
    args = parser.parse_args()
    if any(aliases(args.output, path) for path in args.reports):
        parser.error("output must not overwrite an input report")
    reports, errors = [], []
    for path in args.reports:
        try:
            reports.append(summarize(path))
        except (OSError, ET.ParseError) as error:
            errors.append({"path": str(path), "error": str(error)})
    counts = Counter()
    groups = Counter()
    for report in reports:
        counts.update(report["counts"])
        groups.update(item["signature"] for item in report["failures"])
    full = {"process_exit": args.process_exit, "counts": dict(counts), "reports": reports,
            "parse_errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(full, indent=2) + "\n")
    compact = {
        "process_exit": args.process_exit,
        "reports": len(reports),
        "counts": dict(counts),
        "suite_errors": sum(len(report["suite_errors"]) for report in reports),
        "failure_signatures": [{"count": count, "signature": text} for text, count in groups.most_common(5)],
        "omitted_signature_groups": max(0, len(groups) - 5),
        "parse_errors": len(errors),
        "details": str(args.output.resolve()),
    }
    print(json.dumps(compact, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
