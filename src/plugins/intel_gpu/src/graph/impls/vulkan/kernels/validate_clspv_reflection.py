# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from pathlib import Path


WORKGROUP_SPEC_IDS = {
    "workgroup_size_x": "0",
    "workgroup_size_y": "1",
    "workgroup_size_z": "2",
}


def fields(row: list[str], prefix_size: int) -> dict[str, str]:
    tail = row[prefix_size:]
    if len(tail) % 2 != 0:
        raise ValueError(f"malformed reflection row: {','.join(row)}")
    return dict(zip(tail[0::2], tail[1::2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CLSPV Vulkan bootstrap ABI")
    parser.add_argument("reflection", type=Path)
    parser.add_argument("stamp", type=Path)
    parser.add_argument("entry_point")
    args = parser.parse_args()

    with args.reflection.open(newline="") as stream:
        rows = list(csv.reader(stream))

    kernel_args: dict[int, dict[str, str]] = {}
    specialization_ids: dict[str, str] = {}
    for row in rows:
        if len(row) >= 4 and row[0] == "kernel" and row[1] == args.entry_point and row[2] == "arg":
            values = fields(row, 4)
            kernel_args[int(values["argOrdinal"])] = values
        elif len(row) >= 2 and row[0] == "spec_constant":
            specialization_ids[row[1]] = fields(row, 2)["spec_id"]

    if set(kernel_args) != {0, 1, 2}:
        raise ValueError(f"{args.entry_point} must expose two buffers followed by one POD argument")
    for ordinal in (0, 1):
        values = kernel_args[ordinal]
        expected = {"argKind": "buffer", "descriptorSet": "0", "binding": str(ordinal)}
        actual = {name: values.get(name) for name in expected}
        if actual != expected:
            raise ValueError(f"argument {ordinal} ABI mismatch: expected {expected}, got {actual}")
    pod_expected = {"argKind": "pod_pushconstant", "offset": "0", "argSize": "4"}
    pod_actual = {name: kernel_args[2].get(name) for name in pod_expected}
    if pod_actual != pod_expected:
        raise ValueError(f"POD argument ABI mismatch: expected {pod_expected}, got {pod_actual}")
    if specialization_ids != WORKGROUP_SPEC_IDS:
        raise ValueError(
            f"work-group specialization ABI mismatch: expected {WORKGROUP_SPEC_IDS}, got {specialization_ids}"
        )

    args.stamp.write_text("CLSPV bootstrap reflection is compatible with the Vulkan compute ABI\n")


if __name__ == "__main__":
    main()
