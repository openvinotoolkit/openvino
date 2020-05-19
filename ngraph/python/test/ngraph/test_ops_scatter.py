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
import numpy as np
import ngraph as ng
from ngraph.impl import Type


def test_scatter_nd_update_props():
    dtype = np.int32
    parameter_r = ng.parameter([1000, 256, 10, 15], dtype=dtype, name="data")
    parameter_i = ng.parameter([25, 125, 3], dtype=dtype, name="indices")
    parameter_u = ng.parameter([25, 125, 15], dtype=dtype, name="updates")

    node = ng.scatter_nd_update(parameter_r, parameter_i, parameter_u)
    assert node.get_type_name() == "ScatterNDUpdate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1000, 256, 10, 15]
    assert node.get_output_element_type(0) == Type.i32


def test_scatter_update_props():
    dtype = np.int8
    parameter_r = ng.parameter([2, 3, 4], dtype=dtype, name="data")
    parameter_i = ng.parameter([2, 1], dtype=dtype, name="indices")
    parameter_u = ng.parameter([2, 2, 1, 4], dtype=dtype, name="updates")
    axis = np.array([1], dtype=np.int8)

    node = ng.scatter_update(parameter_r, parameter_i, parameter_u, axis)
    assert node.get_type_name() == "ScatterUpdate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 3, 4]
    assert node.get_output_element_type(0) == Type.i8


def test_scatter_update_elements_props():
    dtype = np.int8
    parameter_r = ng.parameter([2, 4, 5, 7], dtype=dtype, name="data")
    parameter_i = ng.parameter([2, 2, 2, 2], dtype=dtype, name="indices")
    parameter_u = ng.parameter([2, 2, 2, 2], dtype=dtype, name="updates")
    axis = np.array([1], dtype=np.int8)

    node = ng.scatter_elements_update(parameter_r, parameter_i, parameter_u, axis)
    assert node.get_type_name() == "ScatterElementsUpdate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 4, 5, 7]
    assert node.get_output_element_type(0) == Type.i8
