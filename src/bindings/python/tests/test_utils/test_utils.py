# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import os
import sys
import numpy as np
import pytest

from pathlib import Path
from platform import processor

import openvino
import openvino.runtime.opset8 as ops
from openvino.runtime import Model, Core, Shape
from openvino.utils import deprecated


def test_compare_models():
    try:
        from openvino.test_utils import compare_models
        model = get_relu_model()
        status, _ = compare_models(model, model)
        assert status
    except RuntimeError:
        print("openvino.test_utils.compare_models is not available")  # noqa: T201


def generate_lib_name(device, full_device_name):
    lib_name = ""
    arch = processor()
    if arch == "x86_64" or "Intel" in full_device_name or device in ["GNA", "VPUX"]:
        lib_name = "openvino_intel_" + device.lower() + "_plugin"
    elif arch != "x86_64" and device == "CPU":
        lib_name = "openvino_arm_cpu_plugin"
    elif device in ["HETERO", "MULTI", "AUTO"]:
        lib_name = "openvino_" + device.lower() + "_plugin"
    return lib_name


def plugins_path(device, full_device_name):
    lib_name = generate_lib_name(device, full_device_name)
    full_lib_name = ""

    if sys.platform == "win32":
        full_lib_name = lib_name + ".dll"
    else:
        full_lib_name = "lib" + lib_name + ".so"

    plugin_xml = f"""<ie>
    <plugins>
        <plugin location="{full_lib_name}" name="CUSTOM">
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


def get_relu_model(input_shape: List[int] = None) -> openvino.runtime.Model:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    param = ops.parameter(input_shape, np.float32, name="data")
    relu = ops.relu(param, name="relu")
    model = Model([relu], [param], "test_model")
    model.get_ordered_ops()[2].friendly_name = "friendly"

    assert model is not None
    return model


def generate_relu_compiled_model(device, input_shape: List[int] = None) -> openvino.runtime.CompiledModel:
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    model = get_relu_model(input_shape)
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


def create_filename_for_test(test_name, is_xml_path=False, is_bin_path=False):
    """Return a tuple with automatically generated paths for xml and bin files.

    :param test_name: Name used in generating.
    :param is_xml_path: True if xml file should be pathlib.Path object, otherwise return string.
    :param is_bin_path: True if bin file should be pathlib.Path object, otherwise return string.
    :return: Tuple with two objects representing xml and bin files.
    """
    python_version = str(sys.version_info.major) + "_" + str(sys.version_info.minor)
    filename = "./" + test_name.replace("test_", "").replace("[", "_").replace("]", "_")
    filename = filename + "_" + python_version
    _xml = Path(filename + ".xml") if is_xml_path else filename + ".xml"
    _bin = Path(filename + ".bin") if is_bin_path else filename + ".bin"
    return (_xml, _bin)
