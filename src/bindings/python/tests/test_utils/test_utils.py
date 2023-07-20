# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import os
import sys
import numpy as np
import pytest

from pathlib import Path

import openvino
import openvino.runtime.opset12 as ops
from openvino.runtime import Model, Core, Shape
from openvino.utils import deprecated


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


def test_compare_models_pass():
    model = get_relu_model()
    assert compare_models(model, model)


def test_compare_models_fail():
    model = get_relu_model()

    changed_model = model.clone()
    changed_model.get_ordered_ops()[0].set_friendly_name("ABC")

    with pytest.raises(RuntimeError) as e:
        _ = compare_models(model, changed_model)
    assert "Not equal op names model_one: data, model_two: ABC." in str(e.value)


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


def get_relu_model(input_shape: List[int] = None, input_dtype=np.float32) -> openvino.runtime.Model:
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
) -> openvino.runtime.CompiledModel:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    model = get_relu_model(input_shape, input_dtype)
    core = Core()
    return core.compile_model(model, device, {})


def generate_model_and_image(device, input_shape: List[int] = None):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    return (generate_relu_compiled_model(device, input_shape), generate_image(input_shape))


def generate_add_model() -> openvino._pyopenvino.Model:
    param1 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)
    return Model(add, [param1, param2], "TestFunction")


def test_deprecation_decorator():
    @deprecated()
    def deprecated_function1(param1, param2=None):
        pass

    @deprecated(version="2025.4")
    def deprecated_function2(param1=None):
        pass

    @deprecated(message="Use another function instead")
    def deprecated_function3():
        pass

    @deprecated(version="2025.4", message="Use another function instead")
    def deprecated_function4():
        pass

    with pytest.warns(DeprecationWarning, match="deprecated_function1 is deprecated"):
        deprecated_function1("param1")
    with pytest.warns(DeprecationWarning, match="deprecated_function2 is deprecated and will be removed in version 2025.4"):
        deprecated_function2(param1=1)
    with pytest.warns(DeprecationWarning, match="deprecated_function3 is deprecated. Use another function instead"):
        deprecated_function3()
    with pytest.warns(DeprecationWarning, match="deprecated_function4 is deprecated and will be removed in version 2025.4. Use another function instead"):
        deprecated_function4()


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
