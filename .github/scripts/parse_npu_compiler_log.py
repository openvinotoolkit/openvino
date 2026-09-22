#!/usr/bin/env python3
"""Extract per-model NPU compilation metrics from a pytest compiler log.

Output structure:

    {
      "<npu_platform>": {
        "<framework>": {
          "<model>": {
            "compilation_memory_usage_kb": <float|null>,
            "compile_net_time_ms": <float|null>
          }
        }
      }
    }

The script merges into an existing output JSON when present, so several
invocations (e.g. the four PyTorch groups, or the TensorFlow
convert_model/read_model steps) accumulate into a single file per NPU
platform.
"""

from __future__ import annotations

import argparse
import json
import re
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


def load_existing(path: Path) -> dict:
    """Return previously collected results, or an empty dict."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, help="NPU platform, e.g. 3720")
    parser.add_argument(
        "--framework",
        required=True,
        choices=["pytorch", "tensorflow", "jax"],
        help="Frontend/framework the parsed models belong to",
    )
    parser.add_argument("--input", required=True, type=Path, help="Compiler log file")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file; merged into if it already exists",
    )
    args = parser.parse_args()

    result = load_existing(args.output)
    platform = result.setdefault(str(args.platform), {})
    models: dict[str, dict[str, object]] = platform.setdefault(args.framework, {})

    current_model: str | None = None

    for raw_line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)

        case = PYTEST_CASE.match(line)
        if case:
            current_model = case.group("model")
            models.setdefault(
                current_model,
                {
                    "compilation_memory_usage_kb": None,
                    "compile_net_time_ms": None,
                },
            )
            continue

        if current_model is None:
            continue

        memory_kb = find_value(MEMORY_PATTERNS, line)
        if memory_kb is not None:
            models[current_model]["compilation_memory_usage_kb"] = memory_kb
            continue

        compile_time_ms = find_value(TIME_PATTERNS, line)
        if compile_time_ms is not None:
            models[current_model]["compile_net_time_ms"] = compile_time_ms

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
