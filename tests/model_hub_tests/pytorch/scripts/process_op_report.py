# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide filename"
    filename = sys.argv[1]
    if os.path.isfile(filename):
        ops_dict = dict()
        with open(filename, 'r') as f:
            for line in f.readlines():
                r = line.split()
                if r[0] in ops_dict:
                    ops_dict[r[0]].append(r[1])
                else:
                    ops_dict[r[0]] = [r[1]]

        with open(filename, 'w') as f:
            for op in sorted(ops_dict.keys()):
                models = ops_dict[op]
                m_str = ', '.join(models)
                f.write(
                    f"{op:<30} appears in {len(models):>2} models: {m_str}\n")
    else:
        print(f"File {filename} doesn't exist.")
