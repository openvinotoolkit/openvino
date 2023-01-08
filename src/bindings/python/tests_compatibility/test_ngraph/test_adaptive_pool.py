# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
import numpy as np

from ngraph.impl import Type

def test_adaptive_avg_pool():
    input_parameter = ng.parameter((2, 3, 7), name="input_data", dtype=np.float32)
    output_shape = ng.constant(np.array([3], dtype=np.int32))

    adaptive_pool_node = ng.adaptive_avg_pool(input_parameter, output_shape)
    assert adaptive_pool_node.get_type_name() == "AdaptiveAvgPool"
    assert adaptive_pool_node.get_output_size() == 1
    assert adaptive_pool_node.get_output_element_type(0) == Type.f32
    assert list(adaptive_pool_node.get_output_shape(0)) == [2, 3, 3]


def test_adaptive_max_pool():
    input_parameter = ng.parameter((2, 3, 7), name="input_data", dtype=np.float32)
    output_shape = ng.constant(np.array([3], dtype=np.int32))

    adaptive_pool_node = ng.adaptive_max_pool(input_parameter, output_shape)
    assert adaptive_pool_node.get_type_name() == "AdaptiveMaxPool"
    assert adaptive_pool_node.get_output_size() == 2
    assert adaptive_pool_node.get_output_element_type(0) == Type.f32
    assert adaptive_pool_node.get_output_element_type(1) == Type.i64
    assert list(adaptive_pool_node.get_output_shape(0)) == [2, 3, 3]
    assert list(adaptive_pool_node.get_output_shape(1)) == [2, 3, 3]
