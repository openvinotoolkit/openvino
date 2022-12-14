# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np

import openvino.runtime as ov


def type_to_ovtype(general_type):
    if isinstance(general_type, ov.Type):
        return general_type
    types_map = {
        np.float32: ov.Type.f32,
        np.float64: ov.Type.f64,
        np.int8: ov.Type.i8,
        np.int16: ov.Type.i16,
        np.int32: ov.Type.i32,
        np.int64: ov.Type.i64,
        np.uint8: ov.Type.u8,
        np.uint16: ov.Type.u16,
        np.uint32: ov.Type.u32,
        np.uint64: ov.Type.u64,
    }
    return types_map[general_type]


def count_ops_of_type(func, op_type):
    count = 0
    for op in func.get_ops():
        if (type(op) is type(op_type)):
            count += 1
    return count
