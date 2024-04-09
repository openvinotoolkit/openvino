# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys

not_supported_ops_with_translators = ["WriteFile"]


def run_in_ci():
    if "CI" in os.environ and os.environ["CI"].lower() == "true":
        return True

    if "TF_BUILD" in os.environ and len(os.environ["TF_BUILD"]):
        return True

    if "JENKINS_URL" in os.environ and len(os.environ["JENKINS_URL"]):
        return True

    return False


if not run_in_ci():
    # execute check only in CI when the code is productized
    exit(0)

if len(sys.argv) < 3:
    error_message = "Run in the following format: check_supported_ops.py op_table.cpp supported_ops.md"
    raise Exception(error_message)

op_table_src = sys.argv[1]
supported_ops_doc = sys.argv[2]

# parse source file on implemented operations
supported_ops = []
pattern = r'"([^"]*)"'
with open(op_table_src, 'rt') as f:
    for line in f.readlines():
        all_operations = re.findall(pattern, line)
        for operation in all_operations:
            if operation not in not_supported_ops_with_translators:
                supported_ops.append(operation)

# parse a document of supported operations
documented_ops = {}
table_line = 0
with open(supported_ops_doc, 'rt') as f:
    for line in f.readlines():
        is_table_line = False
        if line.count('|') == 4:
            table_line += 1
            is_table_line = True
        if table_line > 2 and is_table_line:
            # skip a table header
            documented_op = line.split('|')[1].strip()
            # remove NEW mark
            documented_op = documented_op.replace('<sup><mark style="background-color: #00FF00">NEW</mark></sup>', '')
            is_supported = False
            if line.split('|')[2].strip() == 'YES':
                is_supported = True
            documented_ops[documented_op] = is_supported

# collect undocumented supported operations
undocumented_supported_ops = []
for supported_op in supported_ops:
    if supported_op in documented_ops and not documented_ops[supported_op]:
        undocumented_supported_ops.append(supported_op)

# collect operations documented as supported by mistake
incorrectly_documented_ops = []
for operation, is_supported in documented_ops.items():
    if is_supported and operation not in supported_ops:
        incorrectly_documented_ops.append(operation)

if len(undocumented_supported_ops) > 0:
    raise Exception('[TensorFlow Frontend] failed: Undocumented Supported Operations = ', undocumented_supported_ops)

if len(incorrectly_documented_ops) > 0:
    raise Exception('[TensorFlow Frontend] failed: Incorrectly Documented Operations = ', incorrectly_documented_ops)
