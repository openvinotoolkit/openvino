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

import os

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from openvino.inference_engine import IECore

from ngraph.impl import Function
from tests.runtime import get_runtime
from tests.test_onnx.utils.onnx_helpers import import_onnx_model


def test_import_onnx_function():
    model_path = os.path.join(os.path.dirname(__file__), "models/add_abc.onnx")
    ie = IECore()
    ie_network = ie.read_network(model=model_path)

    capsule = ie_network._get_function_capsule()
    ng_function = Function.from_capsule(capsule)

    dtype = np.float32
    value_a = np.array([1.0], dtype=dtype)
    value_b = np.array([2.0], dtype=dtype)
    value_c = np.array([3.0], dtype=dtype)

    runtime = get_runtime()
    computation = runtime.computation(ng_function)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([6], dtype=dtype))


def test_simple_graph():
    node1 = make_node("Add", ["A", "B"], ["X"], name="add_node1")
    node2 = make_node("Add", ["X", "C"], ["Y"], name="add_node2")
    graph = make_graph(
        [node1, node2],
        "test_graph",
        [
            make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1]),
            make_tensor_value_info("B", onnx.TensorProto.FLOAT, [1]),
            make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1]),
        ],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])],
    )
    model = make_model(graph, producer_name="ngraph ONNX Importer")

    ng_model_function = import_onnx_model(model)

    runtime = get_runtime()
    computation = runtime.computation(ng_model_function)
    assert np.array_equal(computation(1, 2, 3)[0], np.array([6.0], dtype=np.float32))
    assert np.array_equal(computation(4, 5, 6)[0], np.array([15.0], dtype=np.float32))
