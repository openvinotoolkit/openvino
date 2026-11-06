# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Layout, PartialShape, Type
from openvino.tools.benchmark.utils.inputs_filling import get_image_tensors
from openvino.tools.benchmark.utils.utils import AppInputInfo

cv2 = pytest.importorskip('cv2')


@pytest.mark.parametrize('layout', ['HWC', 'CHW', 'NHWC', 'NCHW'])
def test_image_loading_with_and_without_batch(tmp_path, layout):
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    path = tmp_path / 'image.png'
    assert cv2.imwrite(str(path), image)
    expected = image.transpose(2, 0, 1) if 'CHW' in layout else image
    if layout.startswith('N'):
        expected = expected[None]
    info = AppInputInfo()
    info.name, info.layout = 'image', Layout(layout)
    info.partial_shape, info.element_type = PartialShape(expected.shape), Type.u8
    tensors = get_image_tensors([str(path)], info, [1])
    assert len(tensors) == 1
    np.testing.assert_array_equal(tensors[0].data, expected)
