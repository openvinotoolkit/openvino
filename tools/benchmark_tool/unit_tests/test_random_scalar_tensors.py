# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Layout, PartialShape, Type
from openvino.tools.benchmark.utils.inputs_filling import fill_tensors_with_random
from openvino.tools.benchmark.utils.utils import AppInputInfo


@pytest.mark.parametrize('element_type', [Type.bf16, Type.u4, Type.i4, Type.f32, Type.i32])
@pytest.mark.parametrize('shape', [[], [1], [2, 3]])
def test_random_tensors_preserve_declared_rank(element_type, shape):
    info = AppInputInfo()
    info.layout = Layout()
    info.partial_shape, info.element_type = PartialShape(shape), element_type
    tensors = fill_tensors_with_random(info)
    assert len(tensors) == 1
    assert list(tensors[0].shape) == shape
    assert tensors[0].element_type == element_type
    repeated = fill_tensors_with_random(info)[0]
    np.testing.assert_array_equal(tensors[0].data, repeated.data)
