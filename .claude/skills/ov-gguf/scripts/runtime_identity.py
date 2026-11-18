# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify expected libraries in a live Linux inference process."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def record_identity(expected, output, pid=None):
    pid = os.getpid() if pid is None else int(pid)
    report = {"pid": pid, "expected": expected, "mapped_libraries": [], "errors": []}
    protected = (
        [entry.get("path") for entry in expected.values() if isinstance(entry, dict)]
        if isinstance(expected, dict) else []
    )
    mapped = {}
    try:
        expected_names = set()
        for name in protected:
            if isinstance(name, (str, os.PathLike)):
                path = Path(name)
                expected_names.update((path.name, path.resolve().name))
        for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            name = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[5])
            deleted = name.endswith(" (deleted)")
            path = Path(name.removesuffix(" (deleted)"))
            if path.name.startswith("libopenvino") or path.name in expected_names:
                mapped[path.resolve()] = deleted
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"cannot discover mapped libraries; refusing report output: {error}") from error
    output_path = Path(output)
    protected.extend(mapped)
    for name in protected:
        if not isinstance(name, (str, os.PathLike)):
            continue
        library_path = Path(name)
        same_file = output_path.resolve() == library_path.resolve()
        if output_path.exists() and library_path.exists():
            same_file = same_file or os.path.samefile(output_path, library_path)
        if same_file:
            raise ValueError("output must not overwrite an expected or mapped library")
    report["mapped_libraries"] = [
        {"path": str(path), "sha256": None, "deleted": deleted}
        for path, deleted in sorted(mapped.items())
    ]
    try:
        if not isinstance(expected, dict):
            raise ValueError("expected library manifest must be an object")
        libraries = {}
        for label, entry in expected.items():
            path = Path(entry["path"])
            digest = entry["sha256"]
            if not path.is_absolute() or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"{label}: expected an absolute path and SHA-256")
            libraries[label] = (path.resolve(strict=True), digest.lower())
        if not libraries:
            raise ValueError("expected library manifest is empty")
        hashes = {}
        for entry in report["mapped_libraries"]:
            path = Path(entry["path"])
            digest = None if entry["deleted"] else hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path] = digest
            entry["sha256"] = digest
        for label, (path, digest) in libraries.items():
            if path not in hashes:
                report["errors"].append(f"{label}: expected library is not mapped: {path}")
            elif hashes[path] != digest:
                report["errors"].append(f"{label}: mapped library hash differs: {path}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        report["errors"].append(str(error))
    report["passed"] = not report["errors"]
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("; ".join(report["errors"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", type=int, default=os.getpid())
    args = parser.parse_args()
    same_file = args.output.resolve() == args.expected.resolve()
    if args.output.exists() and args.expected.exists():
        same_file = same_file or os.path.samefile(args.output, args.expected)
    if same_file:
        parser.error("output must not overwrite the expected library manifest")
    record_identity(json.loads(args.expected.read_text()), args.output, args.pid)


if __name__ == "__main__":
    main()
