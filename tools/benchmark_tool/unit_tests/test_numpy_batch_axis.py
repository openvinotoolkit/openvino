# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Layout, PartialShape, Type
from openvino.tools.benchmark.utils.inputs_filling import get_numpy_tensors
from openvino.tools.benchmark.utils.utils import AppInputInfo


@pytest.mark.parametrize('shape,layout', [([2, 3], 'NC'), ([3, 2], 'CN'), ([3, 2, 4], 'C...N')])
def test_numpy_batch_axis(tmp_path, shape, layout):
    info = AppInputInfo()
    info.name, info.layout = 'data', Layout(layout)
    info.partial_shape, info.element_type = PartialShape(shape), Type.f32
    axis = info.layout.get_index_by_name('N') % len(shape)
    arrays = [np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 100 * i for i in range(shape[axis])]
    paths = [tmp_path / f'{i}.npy' for i in range(len(arrays))]
    for path, array in zip(paths, arrays):
        np.save(path, array)
    tensors = get_numpy_tensors([str(path) for path in paths], info, [shape[axis]])
    expected = np.stack([np.take(array, i, axis=axis) for i, array in enumerate(arrays)], axis=axis)
    np.testing.assert_array_equal(tensors[0].data, expected)


@pytest.mark.parametrize('layout,shape', [('NC', [1, 3]), ('CN', [3, 1]), ('C...N', [3, 1])])
def test_numpy_dynamic_shape_inserts_batch_at_layout_axis(tmp_path, layout, shape):
    info = AppInputInfo()
    info.name, info.layout = 'data', Layout(layout)
    info.partial_shape, info.element_type = PartialShape(shape), Type.f32
    info.partial_shape[info.layout.get_index_by_name('N') % len(shape)] = -1
    path = tmp_path / 'input.npy'
    data = np.array([2, 4, 6], dtype=np.float32)
    np.save(path, data)
    tensor = get_numpy_tensors([str(path)], info, [1])[0]
    np.testing.assert_array_equal(tensor.data, data.reshape(shape))
    assert list(info.data_shapes[0]) == shape


def test_numpy_missing_nonbatch_axis_is_not_invented(tmp_path):
    info = AppInputInfo()
    info.name, info.layout = 'data', Layout('CH')
    info.partial_shape, info.element_type = PartialShape([1, -1]), Type.f32
    path = tmp_path / 'input.npy'
    np.save(path, np.arange(3, dtype=np.float32))
    with pytest.raises(Exception, match='not compatible'):
        get_numpy_tensors([str(path)], info, [1])
