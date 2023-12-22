# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import os

import numpy as np
import ngraph as ng
import pytest
from openvino.inference_engine import IECore

from tests_compatibility.runtime import get_runtime


@pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                   reason='Ticket - 122712')
def test_import_onnx_with_external_data():
    model_path = os.path.join(os.path.dirname(__file__), "models/external_data.onnx")
    ie = IECore()
    ie_network = ie.read_network(model=model_path)

    ng_function = ng.function_from_cnn(ie_network)

    dtype = np.float32
    value_a = np.array([1.0, 3.0, 5.0], dtype=dtype)
    value_b = np.array([3.0, 5.0, 1.0], dtype=dtype)
    # third input [5.0, 1.0, 3.0] read from external file

    runtime = get_runtime()
    computation = runtime.computation(ng_function)
    result = computation(value_a, value_b)
    assert np.allclose(result, np.array([3.0, 3.0, 3.0], dtype=dtype))
