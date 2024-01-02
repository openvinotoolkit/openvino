# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from openvino.runtime import opset13 as opset

import openvino as ov
import pytest


@pytest.mark.parametrize(("ov_type", "numpy_dtype"), [
    (ov.Type.f32, np.float32),
    (ov.Type.f64, np.float64),
    (ov.Type.f16, np.float16),
])
def test_float_to_nf4_convert(ov_type, numpy_dtype):
    data = np.linspace(-1.5, 1.5, num=41, dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov.Type.nf4, name="nf4_constant")
    convert = opset.convert(compressed_const, data.dtype)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    uniq = []
    for res_val in result:
        if res_val not in uniq:
            uniq.append(res_val)
    uniq = np.array(uniq)

    assert len(uniq) == 16

    target = [-1.0, -0.6961928009986877, -0.5250730514526367,
              -0.39491748809814453, -0.28444138169288635,
              -0.18477343022823334, -0.09105003625154495,
              0.0, 0.07958029955625534, 0.16093020141124725,
              0.24611230194568634, 0.33791524171829224,
              0.44070982933044434, 0.5626170039176941,
              0.7229568362236023, 1.0]
    target = np.array(target)

    diff = np.max(np.abs(target - uniq))

    assert diff < 0.001
