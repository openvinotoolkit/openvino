# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Layout, PartialShape, Type
from openvino.tools.benchmark.utils.inputs_filling import get_binary_tensors
from openvino.tools.benchmark.utils.utils import AppInputInfo


@pytest.mark.parametrize('shape,layout', [([2, 3], 'NC'), ([3, 2], 'CN'), ([2, 2, 3], 'NCH')])
def test_binary_batches_preserve_each_element(tmp_path, shape, layout):
    info = AppInputInfo()
    info.name, info.layout = 'data', Layout(layout)
    info.partial_shape, info.element_type = PartialShape(shape), Type.f32
    arrays = [np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 10 * i for i in range(2)]
    paths = [tmp_path / f'{i}.bin' for i in range(2)]
    for array, path in zip(arrays, paths):
        array.tofile(path)
    tensors = get_binary_tensors([str(path) for path in paths], info, [2])
    axis = layout.index('N')
    expected = np.stack([np.take(array, i, axis=axis) for i, array in enumerate(arrays)], axis=axis)
    np.testing.assert_array_equal(tensors[0].data, expected)
    assert list(tensors[0].shape) == shape


def test_binary_without_batch_preserves_shape_and_values(tmp_path):
    info = AppInputInfo()
    info.name, info.layout = 'data', Layout('CH')
    info.partial_shape, info.element_type = PartialShape([2, 3]), Type.i32
    data = np.arange(6, dtype=np.int32).reshape(2, 3)
    path = tmp_path / 'input.bin'
    data.tofile(path)
    tensor = get_binary_tensors([str(path)], info, [1])[0]
    np.testing.assert_array_equal(tensor.data, data)


def test_binary_rejects_wrong_file_size(tmp_path):
    info = AppInputInfo()
    info.name, info.layout = 'data', Layout('NC')
    info.partial_shape, info.element_type = PartialShape([1, 3]), Type.f32
    path = tmp_path / 'short.bin'
    np.array([1, 2], dtype=np.float32).tofile(path)
    with pytest.raises(Exception, match='model expects'):
        get_binary_tensors([str(path)], info, [1])
