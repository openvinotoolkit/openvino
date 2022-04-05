# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from openvino.inference_engine import IECore

import ngraph as ng
from ngraph.impl import Function


def np_dtype_to_tensor_type(data_type: np.dtype) -> int:
    """Return TensorProto type for provided numpy dtype.

    :param data_type: Numpy data type object.
    :return: TensorProto.DataType enum value for corresponding type.
    """
    return NP_TYPE_TO_TENSOR_TYPE[data_type]


def import_onnx_model(model: onnx.ModelProto) -> Function:
    onnx.checker.check_model(model)
    model_byte_string = model.SerializeToString()

    ie = IECore()
    ie_network = ie.read_network(model=model_byte_string, weights=b"", init_from_buffer=True)

    ng_function = ng.function_from_cnn(ie_network)
    return ng_function
