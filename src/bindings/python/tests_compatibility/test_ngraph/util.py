# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def count_ops_of_type(func, op_type):
    count = 0
    for op in func.get_ops():
        if (type(op) is type(op_type)):
            count += 1
    return count
