# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
from ngraph.impl import Type

import numpy as np


def test_roll():
    input = np.reshape(np.arange(10, dtype=np.int64), (2, 5))
    input_tensor = ng.constant(input)
    input_shift = ng.constant(np.array([-10, 7], dtype=np.int32))
    input_axes = ng.constant(np.array([-1, 0], dtype=np.int32))

    roll_node = ng.roll(input_tensor, input_shift, input_axes)
    assert roll_node.get_output_size() == 1
    assert roll_node.get_type_name() == "Roll"
    assert list(roll_node.get_output_shape(0)) == [2, 5]
    assert roll_node.get_output_element_type(0) == Type.i64
