# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
import numpy as np
from sys import platform

import openvino as ov
import openvino.runtime.opset13 as ops


def get_dynamic_model():
    param = ops.parameter(ov.PartialShape([-1, -1]), name="input")
    return ov.Model(ops.relu(param), [param])


def get_model(input_shape = None, input_dtype=np.float32) -> ov.Model:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ops.parameter(input_shape, input_dtype, name="input")
    relu = ops.relu(param, name="relu")
    relu.output(0).get_tensor().set_names({"result"})
    model = ov.Model([relu], [param], "test_model")

    assert model is not None
    return model


def get_advanced_model():
    data = ops.parameter(ov.Shape([1, 3, 64, 64]), ov.Type.f32, name="input")
    data1 = ops.parameter(ov.Shape([1, 3, 64, 64]), ov.Type.f32, name="input2")
    axis_const = ops.constant(1, dtype=ov.Type.i64)
    split = ops.split(data, axis_const, 3)
    relu = ops.relu(split.output(1))
    relu.output(0).get_tensor().set_names({"result"})
    return ov.Model([split.output(0), relu.output(0), split.output(2)], [data, data1], "model")


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


def get_path_to_model(input_shape = None):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    path_to_xml = tempfile.NamedTemporaryFile(suffix="_model.xml").name
    model = get_model(input_shape)
    ov.serialize(model, path_to_xml)
    return path_to_xml


def get_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    return temp_dir
