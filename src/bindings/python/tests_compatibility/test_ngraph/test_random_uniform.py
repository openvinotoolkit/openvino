# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
from ngraph.impl import Type

import numpy as np


def test_random_uniform():
    input_tensor = ng.constant(np.array([2, 4, 3], dtype=np.int32))
    min_val = ng.constant(np.array([-2.7], dtype=np.float32))
    max_val = ng.constant(np.array([3.5], dtype=np.float32))

    random_uniform_node = ng.random_uniform(input_tensor, min_val, max_val,
                                            output_type="f32", global_seed=7461,
                                            op_seed=1546)
    random_uniform_node = ng.random_uniform(input_tensor, min_val, max_val,
                                             output_type="f32", global_seed=7461,
                                             op_seed=1546)
    assert random_uniform_node.get_output_size() == 1
    assert random_uniform_node.get_type_name() == "RandomUniform"
    assert random_uniform_node.get_output_element_type(0) == Type.f32
    assert list(random_uniform_node.get_output_shape(0)) == [2, 4, 3]
