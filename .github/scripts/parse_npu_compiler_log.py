#!/usr/bin/env python3
"""Extract per-model NPU compilation metrics from a pytest compiler log.

Output structure:

    {
      "<npu_platform>": {
        "<framework>": {
          "<test_type>": {
            "<model>": {
              "compilation_memory_usage_kb": <float|null>,
              "compile_net_time_ms": <float|null>
            }
          }
        }
      }
    }

The script merges into an existing output JSON when present, so several
invocations (e.g. the four PyTorch groups, or the TensorFlow
convert_model/read_model steps) accumulate into a single file per NPU
platform without overwriting models that appear in multiple suites.

Only the namespaced schema above is supported for merging. If the target
JSON still contains the legacy framework -> model -> metrics layout for a
framework, the script exits with an error instead of silently mixing the
two formats.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC_KEYS = ("compilation_memory_usage_kb", "compile_net_time_ms")
TEST_TYPES = {
    "tensorflow": {"convert_model", "read_model"},
    "jax": {"jax"},
    "pytorch": {"pt_groupA", "pt_groupB", "pt_groupC", "pt_groupD"},
}

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


def default_metrics() -> dict[str, float | None]:
    return {key: None for key in METRIC_KEYS}


def is_metrics_dict(value: object) -> bool:
    return isinstance(value, dict) and set(value).issubset(METRIC_KEYS) and bool(value)


def get_framework_bucket(
    result: dict,
    platform_name: str,
    framework_name: str,
    valid_test_types: set[str],
) -> dict[str, dict[str, object]]:
    platform = result.setdefault(platform_name, {})
    if not isinstance(platform, dict):
        raise SystemExit(f"Existing platform bucket for '{platform_name}' is not a JSON object")

    framework = platform.setdefault(framework_name, {})
    if not isinstance(framework, dict):
        raise SystemExit(f"Existing framework bucket for '{framework_name}' is not a JSON object")

    for bucket_name, models in framework.items():
        if bucket_name not in valid_test_types:
            raise SystemExit(
                f"Existing output has an unexpected test-type bucket '{framework_name}/{bucket_name}'. "
                "Delete the output file and rerun all parser steps to regenerate it."
            )
        if not isinstance(models, dict):
            raise SystemExit(f"Existing test-type bucket for '{framework_name}/{bucket_name}' is not a JSON object")
        if any(key in METRIC_KEYS for key in models):
            raise SystemExit(
                f"Existing output uses the legacy non-namespaced schema for framework '{framework_name}'. "
                "Delete the output file and rerun all parser steps to regenerate it."
            )
        if any(not is_metrics_dict(metrics) for metrics in models.values()):
            raise SystemExit(f"Existing test-type bucket for '{framework_name}/{bucket_name}' has an invalid model-metrics shape")

    return framework


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, help="NPU platform, e.g. 3720")
    parser.add_argument(
        "--framework",
        required=True,
        choices=["pytorch", "tensorflow", "jax"],
        help="Frontend/framework the parsed models belong to",
    )
    parser.add_argument(
        "--test-type",
        required=True,
        help="Stable test namespace within the framework, e.g. convert_model, read_model, jax, pt_groupA",
    )
    parser.add_argument("--input", required=True, type=Path, help="Compiler log file")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file; merged into if it already exists",
    )
    args = parser.parse_args()

    valid_test_types = TEST_TYPES[args.framework]
    if args.test_type not in valid_test_types:
        raise SystemExit(
            f"Unsupported test type '{args.test_type}' for framework '{args.framework}'. "
            f"Expected one of: {', '.join(sorted(valid_test_types))}."
        )

    result = load_existing(args.output)
    framework = get_framework_bucket(result, str(args.platform), args.framework, valid_test_types)
    models: dict[str, dict[str, object]] | None = None
    current_model: str | None = None

    for raw_line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)

        case = PYTEST_CASE.match(line)
        if case:
            models = framework.setdefault(args.test_type, {})
            if not isinstance(models, dict):
                raise SystemExit(f"Existing test-type bucket for '{args.framework}/{args.test_type}' is not a JSON object")
            current_model = case.group("model")
            models.setdefault(current_model, default_metrics())
            continue

        if current_model is None or models is None:
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
