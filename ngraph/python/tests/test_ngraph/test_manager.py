# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
# flake8: noqa
import json

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Function, PartialShape, Shape
from ngraph.impl.passes import Manager
from tests.test_ngraph.util import count_ops_of_type
from ngraph.utils.types import (
    NodeInput,
    NumericData,
    NumericType,
    ScalarData,
    TensorShape,
    as_node,
    as_nodes,
    get_dtype,
    get_element_type,
    get_element_type_str,
    make_constant_node,
)


def test_constant_folding():
    node_constant = ng.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ng.ceiling(node_constant)
    func = Function(node_ceil, [], "TestFunction")

    assert count_ops_of_type(func, node_ceil) == 1
    assert count_ops_of_type(func, node_constant) == 1

    pass_manager = Manager()
    pass_manager.register_pass("ConstantFolding")
    pass_manager.run_passes(func, None)

    assert count_ops_of_type(func, node_ceil) == 0 
    assert count_ops_of_type(func, node_constant) == 1

    # print(as_node(func.get_results()[0]))
    # new_const = ng.constant(as_node(func.get_results()[0]))
    # print(new_const.get_vector())
    # ASSERT_TRUE(new_const);
    # auto values_out = new_const->get_vector<float>();

    # assert f.get_ops() == 0
    # assert shape_out == shape_expected
