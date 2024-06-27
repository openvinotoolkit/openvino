# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from pathlib import Path
from openvino.utils import deprecated, get_cmake_path
from tests.utils.helpers import compare_models, get_relu_model


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
