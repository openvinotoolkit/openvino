# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
from openvino.runtime import Core
import shutil
import pytest
from tests.runtime import get_runtime


external_data_model_current_folder_path = "external_data.onnx"
external_data_model_full_path = os.path.join(os.path.dirname(__file__), "models", external_data_model_current_folder_path)
external_data_current_folder_path = "data/tensor.data"
external_data_full_path = os.path.join(os.path.dirname(__file__), "models", external_data_current_folder_path)


def setup_module():
    shutil.copyfile(external_data_model_full_path, external_data_model_current_folder_path)
    os.mkdir("data")
    shutil.copyfile(external_data_full_path, external_data_current_folder_path)


def teardown_module():
    os.remove(external_data_model_current_folder_path)
    os.remove(external_data_current_folder_path)
    os.rmdir("data")


@pytest.mark.parametrize("model_path", [external_data_model_full_path, external_data_model_current_folder_path])
def test_import_onnx_with_external_data(model_path: str):
    core = Core()
    model = core.read_model(model=model_path)

    dtype = np.float32
    value_a = np.array([1.0, 3.0, 5.0], dtype=dtype)
    value_b = np.array([3.0, 5.0, 1.0], dtype=dtype)
    # third input [5.0, 1.0, 3.0] read from external file

    runtime = get_runtime()
    computation = runtime.computation(model)
    result = computation(value_a, value_b)
    assert np.allclose(result, np.array([3.0, 3.0, 3.0], dtype=dtype))
