# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def count_ops_of_type(model, op_type):
    count = 0
    for op in model.get_ops():
        if (type(op) is type(op_type)):
            count += 1
    return count
