# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from onnx import helper

from tests.tests_python.utils import run_model


def test_resize_with_empty_axes_is_noop():
    input_data = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    scales = np.array([], dtype=np.float32)

    node = helper.make_node("Resize", ["x", "", "scales"], ["y"], axes=[])
    graph = helper.make_graph(
        [node],
        "resize_empty_axes",
        [
            helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, input_data.shape),
            helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, scales.shape),
        ],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, input_data.shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    result = run_model(model, [input_data, scales])[0]
    np.testing.assert_array_equal(result, input_data)
