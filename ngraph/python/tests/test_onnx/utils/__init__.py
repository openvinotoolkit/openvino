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

from string import ascii_uppercase
from typing import Any, Dict, Iterable, List, Optional, Text

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

import tests
from tests.runtime import get_runtime
from tests.test_onnx.utils.onnx_backend import OpenVinoOnnxBackend
from tests.test_onnx.utils.onnx_helpers import import_onnx_model


def xfail_test(reason="Mark the test as expected to fail", strict=True):
    return pytest.mark.xfail(reason=reason, strict=strict)


skip_segfault = pytest.mark.skip(reason="Segmentation fault error")
xfail_issue_35893 = xfail_test(reason="ValueError: could not broadcast input array")
xfail_issue_35911 = xfail_test(reason="Assertion error: Pad model mismatch error")
xfail_issue_35912 = xfail_test(reason="RuntimeError: Error of validate layer: B with type: "
                                      "Pad. Cannot parse parameter pads_end  from IR for layer B. "
                                      "Value -1,0 cannot be casted to int.")
xfail_issue_35914 = xfail_test(reason="IndexError: too many indices for array: "
                                      "array is 0-dimensional, but 1 were indexed")
xfail_issue_35915 = xfail_test(reason="RuntimeError: Eltwise node with unsupported combination "
                                      "of input and output types")
xfail_issue_35916 = xfail_test(reason="RuntimeError: Unsupported input dims count for layer Z")
xfail_issue_35917 = xfail_test(reason="RuntimeError: Unsupported input dims count for "
                                      "layer MatMul")
xfail_issue_35918 = xfail_test(reason="onnx.onnx_cpp2py_export.checker.ValidationError: "
                                      "Mismatched attribute type in 'test_node : alpha'")
xfail_issue_35921 = xfail_test(reason="ValueError - shapes mismatch in gemm")

xfail_issue_35923 = xfail_test(reason="RuntimeError: PReLU without weights is not supported")
xfail_issue_35924 = xfail_test(reason="Assertion error - elu results mismatch")
unstrict_xfail_issue_35925 = xfail_test(reason="Assertion error - reduction ops results mismatch",
                                        strict=False)
strict_xfail_issue_35925 = xfail_test(reason="Assertion error - reduction ops results mismatch")
xfail_issue_35926 = xfail_test(reason="RuntimeError: [NOT_IMPLEMENTED] Input image format I64 is "
                                      "not supported yet...")
xfail_issue_35927 = xfail_test(reason="RuntimeError: B has zero dimension that is not allowable")
xfail_issue_35929 = xfail_test(reason="RuntimeError: Incorrect precision f64!")
xfail_issue_34323 = xfail_test(reason="RuntimeError: data [value] doesn't exist")
xfail_issue_35930 = xfail_test(reason="onnx.onnx_cpp2py_export.checker.ValidationError: "
                                      "Required attribute 'to' is missing.")
xfail_issue_35932 = xfail_test(reason="Assertion error - logsoftmax results mismatch")
xfail_issue_36437 = xfail_test(reason="RuntimeError: Cannot find blob with name: y")


def run_node(onnx_node, data_inputs, **kwargs):
    # type: (onnx.NodeProto, List[np.ndarray], Dict[Text, Any]) -> List[np.ndarray]
    """
    Convert ONNX node to ngraph node and perform computation on input data.

    :param onnx_node: ONNX NodeProto describing a computation node
    :param data_inputs: list of numpy ndarrays with input data
    :return: list of numpy ndarrays with computed output
    """
    OpenVinoOnnxBackend.backend_name = tests.BACKEND_NAME
    return OpenVinoOnnxBackend.run_node(onnx_node, data_inputs, **kwargs)


def run_model(onnx_model, data_inputs):
    # type: (onnx.ModelProto, List[np.ndarray]) -> List[np.ndarray]
    """
    Convert ONNX model to an ngraph model and perform computation on input data.

    :param onnx_model: ONNX ModelProto describing an ONNX model
    :param data_inputs: list of numpy ndarrays with input data
    :return: list of numpy ndarrays with computed output
    """
    ng_model_function = import_onnx_model(onnx_model)
    runtime = get_runtime()
    computation = runtime.computation(ng_model_function)
    return computation(*data_inputs)


def get_node_model(op_type, *input_data, opset=1, num_outputs=1, **node_attributes):
    # type: (str, *Any, Optional[int], Optional[int], **Any) -> onnx.ModelProto
    """Generate model with single requested node.

    Input and output Tensor data type is the same.

    :param op_type: The ONNX node operation.
    :param input_data: Optional list of input arguments for node.
    :param opset: The ONNX operation set version to use. Default to 4.
    :param num_outputs: The number of node outputs.
    :param node_attributes: Optional dictionary of node attributes.
    :return: Generated model with single node for requested ONNX operation.
    """
    node_inputs = [np.array(data) for data in input_data]
    num_inputs = len(node_inputs)
    node_input_names = [ascii_uppercase[idx] for idx in range(num_inputs)]
    node_output_names = [ascii_uppercase[num_inputs + idx] for idx in range(num_outputs)]
    onnx_node = make_node(op_type, node_input_names, node_output_names, **node_attributes)

    input_tensors = [
        make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
        for name, value in zip(onnx_node.input, node_inputs)
    ]
    output_tensors = [
        make_tensor_value_info(name, onnx.TensorProto.FLOAT, ()) for name in onnx_node.output
    ]  # type: ignore

    graph = make_graph([onnx_node], "compute_graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="Ngraph ONNX Importer")
    model.opset_import[0].version = opset
    return model


def all_arrays_equal(first_list, second_list):
    # type: (Iterable[np.ndarray], Iterable[np.ndarray]) -> bool
    """
    Check that all numpy ndarrays in `first_list` are equal to all numpy ndarrays in `second_list`.

    :param first_list: iterable containing numpy ndarray objects
    :param second_list: another iterable containing numpy ndarray objects
    :return: True if all ndarrays are equal, otherwise False
    """
    return all(map(lambda pair: np.array_equal(*pair), zip(first_list, second_list)))
