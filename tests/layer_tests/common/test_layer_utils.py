# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
import openvino.opset13 as ops
import pytest

from common.layer_utils import InferAPI


@pytest.mark.parametrize('shape', [[2, 3], [-1, 3]])
@pytest.mark.parametrize('input_names,expected_names', [
    (['y', 'x'], ['x', 'y']),
    (None, ['output_0', 'output_1']),
    (['unknown'], ['output_0', 'output_1']),
])
def test_get_inputs_info_with_tensor_aliases(tmp_path, shape, input_names, expected_names):
    x = ops.parameter(shape, ov.Type.f32, name='x')
    y = ops.parameter(shape, ov.Type.f32, name='y')
    x.output(0).get_tensor().set_names({'x', 'output_0'})
    y.output(0).get_tensor().set_names({'y', 'output_1'})
    model = ov.Model([x, y], [x, y])
    model_path = tmp_path / 'model.xml'
    weights_path = tmp_path / 'model.bin'
    ov.serialize(model, model_path, weights_path)

    engine = InferAPI(model_path, weights_path, 'CPU')
    inputs_info = engine.get_inputs_info('FP32', input_names)

    assert list(inputs_info) == expected_names
    for actual_shape in inputs_info.values():
        assert ov.PartialShape(actual_shape) == ov.PartialShape(shape)
