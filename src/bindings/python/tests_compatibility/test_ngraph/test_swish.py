# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ngraph as ng
from ngraph.impl import Shape, Type


def test_swish_props_with_beta():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")
    beta = ng.parameter(Shape([]), dtype=float_dtype, name="beta")

    node = ng.swish(data, beta)
    assert node.get_type_name() == "Swish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_swish_props_without_beta():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ng.swish(data)
    assert node.get_type_name() == "Swish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32
