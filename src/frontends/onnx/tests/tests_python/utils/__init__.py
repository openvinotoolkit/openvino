# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from string import ascii_uppercase
from typing import Any, Dict, Iterable, List, Optional, Text

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

import tests
from tests.runtime import get_runtime
from tests.tests_python.utils.onnx_backend import OpenVinoOnnxBackend
from tests.tests_python.utils.onnx_helpers import import_onnx_model


def run_node(onnx_node, data_inputs, **kwargs):
    # type: (onnx.NodeProto, List[np.ndarray], Dict[Text, Any]) -> List[np.ndarray]
    """Convert ONNX node to a graph node and perform computation on input data.

    :param onnx_node: ONNX NodeProto describing a computation node
    :param data_inputs: list of numpy ndarrays with input data
    :return: list of numpy ndarrays with computed output
    """
    OpenVinoOnnxBackend.backend_name = tests.BACKEND_NAME
    return OpenVinoOnnxBackend.run_node(onnx_node, data_inputs, **kwargs)


def run_model(onnx_model, data_inputs):
    # type: (onnx.ModelProto, List[np.ndarray]) -> List[np.ndarray]
    """Convert ONNX model to a graph model and perform computation on input data.

    :param onnx_model: ONNX ModelProto describing an ONNX model
    :param data_inputs: list of numpy ndarrays with input data
    :return: list of numpy ndarrays with computed output
    """
    graph_model = import_onnx_model(onnx_model)
    runtime = get_runtime()
    computation = runtime.computation(graph_model)
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
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")
    model.opset_import[0].version = opset
    return model


def all_arrays_equal(first_list, second_list):
    # type: (Iterable[np.ndarray], Iterable[np.ndarray]) -> bool
    """Check that all numpy ndarrays in `first_list` are equal to all numpy ndarrays in `second_list`.

    :param first_list: iterable containing numpy ndarray objects
    :param second_list: another iterable containing numpy ndarray objects
    :return: True if all ndarrays are equal, otherwise False
    """
    return all(map(lambda pair: np.array_equal(*pair), zip(first_list, second_list)))  # noqa: C417
