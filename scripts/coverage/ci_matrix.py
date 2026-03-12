#!/usr/bin/env python3

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
LIST_TESTS_CMD = ["python3", str(WORKSPACE / "scripts" / "coverage" / "coverage.py"), "list-tests"]


LANE_DEFS = {
    "cpu": {
        "lane": "cpu",
        "test_profile": "cpu",
        "cpp_runner": '["ubuntu-latest-64-cores"]',
        "python_runner": '["ubuntu-latest-8-cores"]',
        "js_runner": '["ubuntu-latest-8-cores"]',
        "requires_gpu_runtime": False,
        "requires_npu_runtime": False,
    },
    "gpu": {
        "lane": "gpu",
        "test_profile": "gpu",
        "cpp_runner": '["self-hosted","Linux","dgpu"]',
        "python_runner": '["self-hosted","Linux","dgpu"]',
        "js_runner": '["self-hosted","Linux","dgpu"]',
        "requires_gpu_runtime": True,
        "requires_npu_runtime": False,
    },
    "npu": {
        "lane": "npu",
        "test_profile": "npu",
        "cpp_runner": '["self-hosted","Linux","npu"]',
        "python_runner": '["self-hosted","Linux","npu"]',
        "js_runner": '["self-hosted","Linux","npu"]',
        "requires_gpu_runtime": False,
        "requires_npu_runtime": True,
    },
}

SELECTION_MAP = {
    "cpu": ["cpu"],
    "cpu_gpu": ["cpu", "gpu"],
    "cpu_gpu_npu": ["cpu", "gpu", "npu"],
}

SUITE_CONFIG = {
    "cpp": {
        "index_key": "cpp_shard_index",
        "count_key": "cpp_shard_count",
        "label_key": "cpp_shard_label",
        "size_key": "cpp_test_count",
        "names_key": "cpp_test_names",
    },
    "python": {
        "index_key": "py_shard_index",
        "count_key": "py_shard_count",
        "label_key": "py_shard_label",
        "size_key": "py_test_count",
        "names_key": "py_test_names",
    },
    "js": {
        "index_key": "js_shard_index",
        "count_key": "js_shard_count",
        "label_key": "js_shard_label",
        "size_key": "js_test_count",
        "names_key": "js_test_names",
    },
}


def _emit_output(**values: str) -> None:
    for key, value in values.items():
        print(f"{key}={value}")


def _list_enabled_tests(*, suite: str, profile: str) -> list[str]:
    output = subprocess.check_output(
        [*LIST_TESTS_CMD, "--suite", suite, "--profile", profile],
        cwd=WORKSPACE,
        text=True,
    )
    rows = csv.DictReader(io.StringIO(output), delimiter="\t")
    return [row["name"] for row in rows if row["enabled"] == "1"]


def resolve_lanes(selection: str) -> None:
    lanes = SELECTION_MAP[selection]
    matrix = {"include": [LANE_DEFS[lane] for lane in lanes]}
    _emit_output(
        matrix=json.dumps(matrix, separators=(",", ":")),
        lanes=",".join(lanes),
    )


def resolve_shards(*, suite: str, base_matrix_json: str, tests_per_job: int) -> None:
    config = SUITE_CONFIG[suite]
    base_matrix = json.loads(base_matrix_json)
    include: list[dict[str, object]] = []

    for lane in base_matrix["include"]:
        enabled = _list_enabled_tests(suite=suite, profile=lane["test_profile"])
        if not enabled:
            continue

        if suite == "js":
            shards = [enabled]
        else:
            shard_size = max(1, tests_per_job)
            shards = [enabled[index : index + shard_size] for index in range(0, len(enabled), shard_size)]

        shard_count = len(shards)
        for shard_index, shard_tests in enumerate(shards, start=1):
            include.append(
                {
                    **lane,
                    config["index_key"]: shard_index,
                    config["count_key"]: shard_count,
                    config["label_key"]: f"{shard_index:02d}-of-{shard_count:02d}",
                    config["size_key"]: len(shard_tests),
                    config["names_key"]: ",".join(shard_tests),
                }
            )

    _emit_output(
        matrix=json.dumps({"include": include}, separators=(",", ":")),
        count=str(len(include)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage CI matrix helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lanes_parser = subparsers.add_parser("resolve-lanes", help="Resolve the workflow lane matrix")
    lanes_parser.add_argument("--selection", choices=sorted(SELECTION_MAP), required=True)

    shard_parser = subparsers.add_parser("resolve-shards", help="Resolve per-suite coverage shard matrix")
    shard_parser.add_argument("--suite", choices=sorted(SUITE_CONFIG), required=True)
    shard_parser.add_argument("--base-matrix-json", required=True)
    shard_parser.add_argument("--tests-per-job", type=int, default=1)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "resolve-lanes":
        resolve_lanes(args.selection)
        return 0
    if args.command == "resolve-shards":
        resolve_shards(suite=args.suite, base_matrix_json=args.base_matrix_json, tests_per_job=args.tests_per_job)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
