# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Layout, PartialShape, Type
from openvino.tools.benchmark.utils.inputs_filling import get_image_info_tensors
from openvino.tools.benchmark.utils.utils import AppInputInfo


@pytest.mark.parametrize('channels', [2, 3, 4])
def test_static_image_info_preserves_width_height_and_scale(channels):
    info = AppInputInfo()
    info.name, info.layout = 'info', Layout('NC')
    info.partial_shape, info.element_type = PartialShape([2, channels]), Type.f32
    tensors = get_image_info_tensors((640, 480), info)
    assert len(tensors) == 1
    expected = np.tile([640, 480] + [1] * (channels - 2), (2, 1))
    np.testing.assert_array_equal(tensors[0].data, expected)


def test_dynamic_image_info_uses_all_shape_groups():
    info = AppInputInfo()
    info.name, info.layout = 'info', Layout('NC')
    info.partial_shape, info.element_type = PartialShape([-1, 3]), Type.f32
    info.data_shapes = [[1, 3], [2, 3], [1, 3]]
    tensors = get_image_info_tensors([(640, 480), (320, 240)], info)
    assert len(tensors) == 3
    for tensor, expected in zip(tensors, [[[640, 480, 1]], [[320, 240, 1]] * 2, [[640, 480, 1]]]):
        np.testing.assert_array_equal(tensor.data, expected)
