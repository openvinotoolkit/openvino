# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime as ov
from openvino.runtime import Shape


def test_get_constant_from_source_success():
    dtype = np.int
    input1 = ov.opset8.parameter(Shape([5, 5]), dtype=dtype, name="input_1")
    input2 = ov.opset8.parameter(Shape([25]), dtype=dtype, name="input_2")
    shape_of = ov.opset8.shape_of(input2, name="shape_of")
    reshape = ov.opset8.reshape(input1, shape_of, special_zero=True)
    folded_const = ov.utils.get_constant_from_source(reshape.input(1).get_source_output())

    assert folded_const is not None
    assert folded_const.get_vector() == [25]


def test_get_constant_from_source_failed():
    dtype = np.int
    input1 = ov.opset8.parameter(Shape([5, 5]), dtype=dtype, name="input_1")
    input2 = ov.opset8.parameter(Shape([1]), dtype=dtype, name="input_2")
    reshape = ov.opset8.reshape(input1, input2, special_zero=True)
    folded_const = ov.utils.get_constant_from_source(reshape.input(1).get_source_output())

    assert folded_const is None
