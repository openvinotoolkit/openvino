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
    try:
        libraries = {}
        for label, entry in expected.items():
            path = Path(entry["path"])
            digest = entry["sha256"]
            if not path.is_absolute() or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"{label}: expected an absolute path and SHA-256")
            libraries[label] = (path.resolve(strict=True), digest.lower())
        if not libraries:
            raise ValueError("expected library manifest is empty")
        expected_names = {path.name for path, _ in libraries.values()}
        mapped = {}
        for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            name = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[5])
            deleted = name.endswith(" (deleted)")
            path = Path(name.removesuffix(" (deleted)"))
            if path.name.startswith("libopenvino") or path.name in expected_names:
                mapped[path.resolve()] = deleted
        hashes = {}
        for path, deleted in sorted(mapped.items()):
            digest = None if deleted else hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path] = digest
            report["mapped_libraries"].append(
                {"path": str(path), "sha256": digest, "deleted": deleted}
            )
        for label, (path, digest) in libraries.items():
            if path not in hashes:
                report["errors"].append(f"{label}: expected library is not mapped: {path}")
            elif hashes[path] != digest:
                report["errors"].append(f"{label}: mapped library hash differs: {path}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        report["errors"].append(str(error))
    report["passed"] = not report["errors"]
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("; ".join(report["errors"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", type=int, default=os.getpid())
    args = parser.parse_args()
    record_identity(json.loads(args.expected.read_text()), args.output, args.pid)


if __name__ == "__main__":
    main()
