# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from sys import platform

import openvino as ov
import openvino.runtime.opset12 as ops

import ngraph as ng
from ngraph.impl import Function


def get_dynamic_model():
    param = ops.parameter(ov.PartialShape([-1, -1]))
    return ov.Model(ops.relu(param), [param])


def get_model(input_shape = None, input_dtype=np.float32) -> ov.Model:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ops.parameter(input_shape, input_dtype, name="data")
    relu = ops.relu(param, name="relu")
    relu.output(0).get_tensor().set_names({"res"})
    model = ov.Model([relu], [param], "test_model")

    assert model is not None
    return model


def get_ngraph_model(input_shape = None, input_dtype=np.float32):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ng.opset11.parameter(input_shape, input_dtype, name="data")
    relu = ng.opset11.relu(param, name="relu")
    model = Function([relu], [param], "test_model")

    assert model is not None
    return model


def get_image(shape = (1, 3, 32, 32), dtype = "float32"):
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)


def get_path_to_extension_library():
    library_path=""
    if platform == "win32":
        library_path="openvino_template_extension.dll"
    else:
        library_path="libopenvino_template_extension.so"
    return library_path


def get_path_to_model(is_old_api=False):
    path_to_xml = tempfile.NamedTemporaryFile(suffix="_model.xml").name
    if is_old_api:
        func = get_ngraph_model(input_dtype=np.int64)
        caps = ng.Function.to_capsule(func)
        net = ie.IENetwork(caps)
        net.serialize(path_to_xml)
    else:
        model = get_model()
        ov.serialize(model, path_to_xml)
    return path_to_xml


def get_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    return temp_dir
