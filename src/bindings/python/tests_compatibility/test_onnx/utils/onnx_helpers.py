# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import onnx
from openvino.inference_engine import IECore

import ngraph as ng
from ngraph.impl import Function


def import_onnx_model(model: onnx.ModelProto) -> Function:
    onnx.checker.check_model(model)
    model_byte_string = model.SerializeToString()

    ie = IECore()
    ie_network = ie.read_network(model=model_byte_string, weights=b"", init_from_buffer=True)

    ng_function = ng.function_from_cnn(ie_network)
    return ng_function
