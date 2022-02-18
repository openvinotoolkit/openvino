# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime.opset8 as ov
from openvino.runtime import Shape, Type


def test_swish_props_with_beta():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")
    beta = ov.parameter(Shape([]), dtype=float_dtype, name="beta")

    node = ov.swish(data, beta)
    assert node.get_type_name() == "Swish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_swish_props_without_beta():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.swish(data)
    assert node.get_type_name() == "Swish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32
