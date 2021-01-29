# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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


def test_constant_folding():
    node_constant = ng.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ng.ceiling(node_constant)
    func = Function(node_ceil, [], "TestFunction")

    assert count_ops_of_type(func, node_ceil) == 1
    assert count_ops_of_type(func, node_constant) == 1

    pass_manager = Manager()
    pass_manager.register_pass("ConstantFolding")
    pass_manager.run_passes(func)

    assert count_ops_of_type(func, node_ceil) == 0
    assert count_ops_of_type(func, node_constant) == 1

    new_const = func.get_results()[0].input(0).get_source_output().get_node()

    values_out = new_const.get_vector()
    values_expected = [0.0, 1.0, 0.0, -2.0, 3.0, 3.0]

    assert np.allclose(values_out, values_expected)
