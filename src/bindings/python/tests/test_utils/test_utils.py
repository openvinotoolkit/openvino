# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import numpy as np
import openvino as ov
from pathlib import Path
from openvino.utils import deprecated, get_cmake_path, make_postponed_constant
from tests.utils.helpers import compare_models, get_relu_model, create_filenames_for_ir


def test_compare_functions():
    try:
        from openvino.test_utils import compare_functions
        model = get_relu_model()
        status, _ = compare_functions(model, model)
        assert status
    except RuntimeError:
        print("openvino.test_utils.compare_functions is not available")  # noqa: T201


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


def test_cmake_file_found(monkeypatch):
    fake_package_path = "/fake/site-packages/openvino"

    def mock_walk(path):
        return [
            ("/fake/site-packages/openvino/dir1", ("subdir",), ("OpenVINOConfig.cmake", "otherfile.txt")),
            ("/fake/site-packages/openvino/dir2", ("subdir",), ("otherfile.txt",)),
        ]

    monkeypatch.setattr(Path, "parent", fake_package_path)
    monkeypatch.setattr(os, "walk", mock_walk)

    result = get_cmake_path()

    assert result == f"{fake_package_path}/dir1"


def test_cmake_file_not_found(monkeypatch):
    fake_package_path = "/fake/site-packages/openvino"

    def mock_walk(path):
        return [
            ("/fake/site-packages/openvino/dir1", ("subdir",), ("otherfile.txt", "OpenVINOConfig")),
            ("/fake/site-packages/openvino/dir2", ("subdir",), ("otherfile.txt", "OpenVINO.cmake")),
        ]

    monkeypatch.setattr(Path, "parent", fake_package_path)
    monkeypatch.setattr(os, "walk", mock_walk)

    result = get_cmake_path()

    assert result == ""


class Maker:
    def __init__(self):
        self.calls_count = 0

    def __call__(self, tensor: ov.Tensor) -> None:
        self.calls_count += 1
        tensor_data = np.array([2, 2, 2, 2], dtype=np.float32).reshape(1, 1, 2, 2)
        ov.Tensor(tensor_data).copy_to(tensor)

    def called_times(self):
        return self.calls_count


def create_model(maker):
    input_shape = ov.Shape([1, 2, 1, 2])
    param_node = ov.opset13.parameter(input_shape, ov.Type.f32, name="data")

    postponned_constant = make_postponed_constant(ov.Type.f32, input_shape, maker)

    add_1 = ov.opset13.add(param_node, postponned_constant)

    const_2 = ov.op.Constant(ov.Type.f32, input_shape, [1, 2, 3, 4])
    add_2 = ov.opset13.add(add_1, const_2)

    return ov.Model(add_2, [param_node], "test_model")


@pytest.fixture
def prepare_ir_paths(request, tmp_path):
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)

    yield xml_path, bin_path

    # IR Files deletion should be done after `Model` is destructed.
    # It may be achieved by splitting scopes (`Model` will be destructed
    # just after test scope finished), or by calling `del Model`
    os.remove(xml_path)
    os.remove(bin_path)


def test_save_postponned_constant(prepare_ir_paths):
    maker = Maker()
    model = create_model(maker)
    assert maker.called_times() == 0

    model_export_file_name, weights_export_file_name = prepare_ir_paths
    ov.save_model(model, model_export_file_name, compress_to_fp16=False)

    assert maker.called_times() == 1


def test_save_postponned_constant_twice(prepare_ir_paths):
    maker = Maker()
    model = create_model(maker)
    assert maker.called_times() == 0

    model_export_file_name, weights_export_file_name = prepare_ir_paths
    ov.save_model(model, model_export_file_name, compress_to_fp16=False)
    assert maker.called_times() == 1
    ov.save_model(model, model_export_file_name, compress_to_fp16=False)
    assert maker.called_times() == 2


def test_serialize_postponned_constant(prepare_ir_paths):
    maker = Maker()
    model = create_model(maker)
    assert maker.called_times() == 0

    model_export_file_name, weights_export_file_name = prepare_ir_paths
    ov.serialize(model, model_export_file_name, weights_export_file_name)
    assert maker.called_times() == 1


def test_infer_postponned_constant():
    maker = Maker()
    model = create_model(maker)
    assert maker.called_times() == 0

    compiled_model = ov.compile_model(model, "CPU")
    assert isinstance(compiled_model, ov.CompiledModel)

    request = compiled_model.create_infer_request()
    input_data = np.ones([1, 2, 1, 2], dtype=np.float32)
    input_tensor = ov.Tensor(input_data)

    results = request.infer({"data": input_tensor})
    assert maker.called_times() == 1

    expected_output = np.array([4, 5, 6, 7], dtype=np.float32).reshape(1, 2, 1, 2)
    assert np.allclose(results[list(results)[0]], expected_output, 1e-4, 1e-4)
