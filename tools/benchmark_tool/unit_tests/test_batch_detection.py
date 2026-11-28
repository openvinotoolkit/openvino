# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino import opset13 as ops
from openvino.tools.benchmark.utils.utils import get_inputs_info


@pytest.mark.parametrize('batch', [1, 2])
def test_existing_batch_is_found_before_input_without_batch(batch):
    image = ops.parameter([batch, 3, 4, 5], name='image')
    auxiliary = ops.parameter([3], name='aux')
    info, reshape = get_inputs_info('', '', 'image[NCHW],aux[C]', batch, '', '',
                                   [image.output(0), auxiliary.output(0)])
    assert not reshape
    assert list(info[0].partial_shape.to_shape()) == [batch, 3, 4, 5]
    assert list(info[1].partial_shape.to_shape()) == [3]


def test_batch_change_still_reshapes():
    image = ops.parameter([1, 3, 4, 5], name='image')
    info, reshape = get_inputs_info('', '', 'image[NCHW]', 2, '', '', [image.output(0)])
    assert reshape
    assert list(info[0].partial_shape.to_shape()) == [2, 3, 4, 5]


def test_batch_without_any_batch_axis_is_rejected():
    auxiliary = ops.parameter([3], name='aux')
    with pytest.raises(RuntimeError, match='batch cannot be set'):
        get_inputs_info('', '', 'aux[C]', 1, '', '', [auxiliary.output(0)])
