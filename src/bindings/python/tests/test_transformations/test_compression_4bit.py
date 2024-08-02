# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from openvino.runtime import opset13 as opset

import openvino as ov
import pytest


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
    ],
)
def test_float_to_nf4_convert(ov_type, numpy_dtype):
    data = np.linspace(-1.5, 1.5, num=41, dtype=numpy_dtype)

    # Compress data to NF4
    compressed_const = opset.constant(data, dtype=ov.Type.nf4, name="nf4_constant")
    # get decompressed data as tested OV type
    decompressed = opset.convert(compressed_const, ov_type)

    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    output = opset.add(parameter, decompressed)
    model = ov.Model([output], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    uniq = []
    for res_val in result:
        if res_val not in uniq:
            uniq.append(res_val)
    uniq = np.array(uniq)

    assert len(uniq) == 16

    target = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
    target = np.array(target)

    diff = np.max(np.abs(target - uniq))

    assert diff < 0.001


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
    ],
)
def test_float_to_f4e2m1_convert(ov_type, numpy_dtype):
    data = np.array(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=numpy_dtype,
    )

    # Compress data
    compressed_const = opset.constant(data, dtype=ov.Type.f4e2m1, name="f4e2m1_constant")
    # get decompressed data as tested OV type
    decompressed = opset.convert(compressed_const, ov_type)

    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    output = opset.add(parameter, decompressed)
    model = ov.Model([output], [parameter])

    compiled_model = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled_model(tensor)[0]

    expected = np.array([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=numpy_dtype)

    assert np.allclose(result, expected)
