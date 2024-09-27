# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import os
import sys
import numpy as np
import base64

from sys import platform
from pathlib import Path

import openvino
from openvino import Model, Core, Shape, Tensor, Type
import openvino.runtime.opset13 as ops


def _compare_models(model_one: Model, model_two: Model, compare_names: bool = True) -> Tuple[bool, str]:  # noqa: C901 the function is too complex
    """Function to compare OpenVINO model (ops names, types and shapes).

    Note that the functions uses get_ordered_ops, so the topological order of ops should be also preserved.

    :param model_one: The first model to compare.
    :param model_two: The second model to compare.
    :param compare_names: Flag to control friendly names checking. Default: True
    :return: Tuple which consists of bool value (True if models are equal, otherwise False)
             and string with the message to reuse for debug/testing purposes. The string value
             is empty when models are equal.
    """
    result = True
    msg = ""

    # Check friendly names of models
    if compare_names and model_one.get_friendly_name() != model_two.get_friendly_name():
        result = False
        msg += "Friendly names of models are not equal "
        msg += f"model_one: {model_one.get_friendly_name()}, model_two: {model_two.get_friendly_name()}.\n"

    model_one_ops = model_one.get_ordered_ops()
    model_two_ops = model_two.get_ordered_ops()

    # Check overall number of operators
    if len(model_one_ops) != len(model_two_ops):
        result = False
        msg += "Not equal number of ops "
        msg += f"model_one: {len(model_one_ops)}, model_two: {len(model_two_ops)}.\n"

    for i in range(len(model_one_ops)):
        op_one_name = model_one_ops[i].get_friendly_name()  # op from model_one
        op_two_name = model_two_ops[i].get_friendly_name()  # op from model_two
        # Check friendly names
        if (compare_names and op_one_name != op_two_name and model_one_ops[i].get_type_name() != "Constant"):
            result = False
            msg += "Not equal op names "
            msg += f"model_one: {op_one_name}, "
            msg += f"model_two: {op_two_name}.\n"
        # Check output sizes
        if model_one_ops[i].get_output_size() != model_two_ops[i].get_output_size():
            result = False
            msg += f"Not equal output sizes of {op_one_name} and {op_two_name}.\n"
        for idx in range(model_one_ops[i].get_output_size()):
            # Check partial shapes of outputs
            op_one_partial_shape = model_one_ops[i].get_output_partial_shape(idx)
            op_two_partial_shape = model_two_ops[i].get_output_partial_shape(idx)
            if op_one_partial_shape != op_two_partial_shape:
                result = False
                msg += f"Not equal op partial shapes of {op_one_name} and {op_two_name} on {idx} index "
                msg += f"model_one: {op_one_partial_shape}, "
                msg += f"model_two: {op_two_partial_shape}.\n"
            # Check element types of outputs
            op_one_element_type = model_one_ops[i].get_output_element_type(idx)
            op_two_element_type = model_two_ops[i].get_output_element_type(idx)
            if op_one_element_type != op_two_element_type:
                result = False
                msg += f"Not equal output element types of {op_one_name} and {op_two_name} on {idx} index "
                msg += f"model_one: {op_one_element_type}, "
                msg += f"model_two: {op_two_element_type}.\n"

    return result, msg


def compare_models(model_one: Model, model_two: Model, compare_names: bool = True):
    """Function to compare OpenVINO model (ops names, types and shapes).

    :param model_one: The first model to compare.
    :param model_two: The second model to compare.
    :param compare_names: Flag to control friendly names checking. Default: True
    :return: True if models are equal, otherwise raise an error with a report of mismatches.
    """
    result, msg = _compare_models(model_one, model_two, compare_names=compare_names)

    if not result:
        raise RuntimeError(msg)

    return result


def plugins_path(device, lib_path):
    plugin_xml = f"""<ie>
    <plugins>
        <plugin location="{lib_path}" name="{device}">
        </plugin>
    </plugins>
    </ie>"""

    with open("plugin_path.xml", "w") as f:
        f.write(plugin_xml)

    plugins_paths = os.path.join(os.getcwd(), "plugin_path.xml")
    return plugins_paths


def generate_image(shape: Tuple = (1, 3, 32, 32), dtype: Union[str, np.dtype] = "float32") -> np.array:
    np.random.seed(42)
    return np.random.rand(*shape).astype(dtype)


def get_model_with_template_extension():
    core = Core()
    ir = bytes(b"""<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Identity" version="extension">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="out_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>""")
    if platform == "win32":
        core.add_extension(library_path="openvino_template_extension.dll")
    else:
        core.add_extension(library_path="libopenvino_template_extension.so")
    return core, core.read_model(ir)


def get_relu_model(input_shape: List[int] = None, input_dtype=np.float32) -> openvino.Model:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ops.parameter(input_shape, input_dtype, name="data")
    relu = ops.relu(param, name="relu")
    model = Model([relu], [param], "test_model")
    model.get_ordered_ops()[2].friendly_name = "friendly"

    assert model is not None
    return model


def generate_relu_compiled_model(
    device,
    input_shape: List[int] = None,
    input_dtype=np.float32,
) -> openvino.CompiledModel:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    model = get_relu_model(input_shape, input_dtype)
    core = Core()
    return core.compile_model(model, device, {})


def encrypt_base64(src):
    return base64.b64encode(bytes(src, "utf-8"))


def decrypt_base64(src):
    return base64.b64decode(bytes(src, "utf-8"))


def generate_relu_compiled_model_with_config(
    device,
    config,
    input_shape: List[int] = None,
    input_dtype=np.float32,
) -> openvino.CompiledModel:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    model = get_relu_model(input_shape, input_dtype)
    core = Core()
    return core.compile_model(model, device, config)


def generate_model_and_image(device, input_shape: List[int] = None):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    return (generate_relu_compiled_model(device, input_shape), generate_image(input_shape))


def generate_add_model(input_shape: List[int] = None, input_dtype=np.float32) -> openvino.Model:
    if input_shape is None:
        input_shape = [2, 1]
    param1 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)
    return Model(add, [param1, param2], "TestModel")


def generate_add_compiled_model(
    device,
    input_shape: List[int] = None,
    input_dtype=np.float32,
) -> openvino.CompiledModel:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    model = generate_add_model(input_shape, input_dtype)
    core = Core()
    return core.compile_model(model, device, {})


def generate_model_with_memory(input_shape, data_type) -> openvino._pyopenvino.Model:
    input_data = ops.parameter(input_shape, name="input_data", dtype=data_type)
    init_val = ops.constant(np.zeros(input_shape), data_type)
    rv = ops.read_value(init_val, "var_id_667", data_type, input_shape)
    add = ops.add(rv, input_data, name="MemoryAdd")
    node = ops.assign(add, "var_id_667")
    res = ops.result(add, "res")
    model = Model(results=[res], sinks=[node], parameters=[input_data], name="TestModel")
    return model


def generate_concat_compiled_model(device, input_shape: List[int] = None, ov_type=Type.f32, numpy_dtype=np.float32):
    if input_shape is None:
        input_shape = [5]

    core = Core()

    params = []
    params += [ops.parameter(input_shape, ov_type)]
    if ov_type == Type.bf16:
        params += [ops.parameter(input_shape, ov_type)]
    else:
        params += [ops.parameter(input_shape, numpy_dtype)]

    model = Model(ops.concat(params, 0), params)
    return core.compile_model(model, device)


def generate_concat_compiled_model_with_data(device, input_shape: List[int] = None, ov_type=Type.f32, numpy_dtype=np.float32):
    if input_shape is None:
        input_shape = [5]

    compiled_model = generate_concat_compiled_model(device, input_shape, ov_type, numpy_dtype)
    request = compiled_model.create_infer_request()
    tensor1 = Tensor(ov_type, input_shape)
    tensor1.data[:] = np.array([6, 7, 8, 9, 0])
    array1 = np.array([1, 2, 3, 4, 5], dtype=numpy_dtype)

    return request, tensor1, array1


def generate_abs_compiled_model_with_data(device, ov_type, numpy_dtype):
    input_shape = [1, 4]
    param = ops.parameter(input_shape, ov_type)
    model = Model(ops.abs(param), [param])
    core = Core()
    compiled_model = core.compile_model(model, device)

    request = compiled_model.create_infer_request()

    tensor1 = Tensor(ov_type, input_shape)
    tensor1.data[:] = np.array([6, -7, -8, 9])

    array1 = np.array([[-1, 2, 5, -3]]).astype(numpy_dtype)

    return compiled_model, request, tensor1, array1


def create_filename_for_test(test_name, tmp_path, is_xml_path=False, is_bin_path=False):
    """Return a tuple with automatically generated paths for xml and bin files.

    :param test_name: Name used in generating.
    :param is_xml_path: True if xml file should be pathlib.Path object, otherwise return string.
    :param is_bin_path: True if bin file should be pathlib.Path object, otherwise return string.
    :return: Tuple with two objects representing xml and bin files.
    """
    python_version = str(sys.version_info.major) + "_" + str(sys.version_info.minor)
    filename = test_name.replace("test_", "").replace("[", "_").replace("]", "_")
    filename = filename + "_" + python_version
    path_to_xml = tmp_path / Path(filename + ".xml")
    path_to_bin = tmp_path / Path(filename + ".bin")
    _xml = path_to_xml if is_xml_path else str(path_to_xml)
    _bin = path_to_bin if is_bin_path else str(path_to_bin)
    return (_xml, _bin)
