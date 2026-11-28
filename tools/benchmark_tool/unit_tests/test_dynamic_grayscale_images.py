# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Layout, PartialShape, Type
from openvino.tools.benchmark.utils.inputs_filling import get_image_tensors
from openvino.tools.benchmark.utils.utils import AppInputInfo

cv2 = pytest.importorskip('cv2')


@pytest.mark.parametrize('layout,shape', [('NHWC', [1, -1, -1, 1]), ('NCHW', [1, 1, -1, -1]),
                                        ('HWC', [-1, -1, 1]), ('CHW', [1, -1, -1])])
def test_dynamic_grayscale_keeps_channel_axis(tmp_path, layout, shape):
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    path = tmp_path / 'image.png'
    assert cv2.imwrite(str(path), image)
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]
    if 'CHW' in layout:
        expected = expected.transpose(2, 0, 1)
    if layout.startswith('N'):
        expected = expected[None]
    info = AppInputInfo()
    info.name, info.layout = 'image', Layout(layout)
    info.partial_shape, info.element_type = PartialShape(shape), Type.u8
    tensors = get_image_tensors([str(path)], info, [1])
    np.testing.assert_array_equal(tensors[0].data, expected)
    assert list(info.data_shapes[0]) == list(expected.shape)
