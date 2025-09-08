# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino import PartialShape, Model, Type
import openvino.opset13 as ops
from openvino.op import Result


def test_result():
    param = ops.parameter(PartialShape([1]), dtype=np.float32, name="param")
    relu1 = ops.relu(param, name="relu1")
    result = Result(relu1.output(0))
    assert result.get_output_element_type(0) == Type.f32
    assert result.get_output_partial_shape(0) == PartialShape([1])
    model = Model([result], [param], "test_model")

    result2 = ops.result(relu1, "res2")
    model.add_results([result2])

    results = model.get_results()
    assert len(results) == 2
    assert results[1].get_output_element_type(0) == Type.f32
    assert results[1].get_output_partial_shape(0) == PartialShape([1])
    model.remove_result(result)
    assert len(model.results) == 1
