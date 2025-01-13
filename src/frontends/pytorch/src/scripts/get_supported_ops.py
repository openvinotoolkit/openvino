# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re


def get_ops_in_specific_map(map_name: str, lines: list) -> list:
    ops = []
    start_found = False
    for line in lines:
        if start_found:
            if "};" in line:
                break
            res = re.findall(r"\s*\{\"(\w*::\w*)\".*", line)
            if len(res) > 0:
                ops.append((res[0], ""))
            else:
                res = re.findall(r"\s*\/\/\s(\w*::\w*)\s\-\s(.*)", line)
                if len(res) > 0:
                    ops.append(res[0])
        if map_name in line:
            start_found = True
    return sorted(ops, key=lambda s: s[0].casefold())


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "..", "op_table.cpp")
    with open(filepath, "r") as f:
        ops = get_ops_in_specific_map("get_supported_ops_ts", f.readlines())
    for op in ops:
        print(f"      {op[0]:<44} {op[1]}")
