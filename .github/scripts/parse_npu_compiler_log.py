#!/usr/bin/env python3
"""Extract per-model NPU compilation metrics from a pytest compiler log."""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# pytest -v output, for example:
# tests/.../test_timm.py::TestTimm::test_timm_precommit[NPU-resnet18-...] PASSED
PYTEST_CASE = re.compile(r"^(?P<nodeid>.+?\.py::.+?\[(?P<model>.+)\])(?:\s|$)")

MEMORY_PATTERNS = [
    re.compile(r"Compilation memory usage:\s*Peak\s*(?P<value>\d+(?:\.\d+)?)\s*KB", re.I),
]

TIME_PATTERNS = [
    re.compile(r"Compile net time:\s*(?P<value>\d+(?:\.\d+)?)\s*ms", re.I),
    re.compile(r"Compile network (?:took|time:)\s*(?P<value>\d+(?:\.\d+)?)\s*ms", re.I),
    re.compile(r"Compile model took\s*(?P<value>\d+(?:\.\d+)?)\s*ms", re.I),
]


def find_value(patterns: list[re.Pattern[str]], line: str) -> float | None:
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            return float(match.group("value"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, help="NPU platform, e.g. 3720")
    parser.add_argument("--input", required=True, type=Path, help="Compiler log file")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    args = parser.parse_args()

    models: OrderedDict[str, dict[str, object]] = OrderedDict()
    current_nodeid: str | None = None

    for raw_line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)

        case = PYTEST_CASE.match(line)
        if case:
            current_nodeid = case.group("nodeid")
            models.setdefault(
                current_nodeid,
                {
                    "model": case.group("model"),
                    "compilation_memory_usage_kb": None,
                    "compile_net_time_ms": None,
                },
            )
            continue

        if current_nodeid is None:
            continue

        memory_kb = find_value(MEMORY_PATTERNS, line)
        if memory_kb is not None:
            models[current_nodeid]["compilation_memory_usage_kb"] = memory_kb
            continue

        compile_time_ms = find_value(TIME_PATTERNS, line)
        if compile_time_ms is not None:
            models[current_nodeid]["compile_net_time_ms"] = compile_time_ms

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({str(args.platform): models}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
