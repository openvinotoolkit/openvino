# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.one_hot import OneHot
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, connect


def generate_nodes(data, axis=-1, depth=4, on_value=1., off_value=0.):
    return {
        'indices': {'Op': 'Parameter', 'value': data, 'shape': int64_array(data.shape)},
        'indices_d': {'kind': 'data', 'value': data, 'shape': int64_array(data.shape)},
        **valued_const_with_data('depth', int64_array(depth)),
        **valued_const_with_data('on_value', float_array(on_value)),
        **valued_const_with_data('off_value', float_array(off_value)),
        **regular_op_with_shaped_data('one_hot', None, {'type': 'OneHot', 'axis': axis, 'Op': 'OneHot'})
    }


edges = [
    *connect('indices:0', 'one_hot:0'),
    *connect('depth:0', 'one_hot:1'),
    *connect('on_value:0', 'one_hot:2'),
    *connect('off_value:0', 'one_hot:3'),
    ('one_hot', 'one_hot_d')
]


class TestOneHotInfer():
    @pytest.mark.parametrize("input_value, exp_value, axis",[
        # 0d input
        (1, [0, 1, 0, 0], -1),
        # 1d input
        ([1, 2], [[0, 1, 0, 0], [0, 0, 1, 0]], -1),
        # 2D input
        ([[1, 2], [3, 4]], [[[0, 1, 0, 0], [0, 0, 1, 0]],
                            [[0, 0, 0, 1], [0, 0, 0, 0]]], -1),
        # 3d input
        ([[[0, 2], [1, 2]], [[2, 1], [3, 0]]],
         [[[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [0, 0, 1, 0]]],
          [[[0, 0, 1, 0], [0, 1, 0, 0]], [[0, 0, 0, 1], [1, 0, 0, 0]]]], -1),
        # 1d input with negative indices
        ([-2, 2], [[0, 0, 1, 0], [0, 0, 1, 0]], -1),
        # check if axis is neither 0 nor -1
        ([[1, 2], [3, 4]], [[[0, 0], [1, 0], [0, 1], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [1, 0]]], 1)
    ])
    def test_infer(self, input_value, exp_value, axis):
        graph = build_graph(generate_nodes(int64_array(input_value), axis), edges)
        onehot_node = Node(graph, 'one_hot')
        OneHot.infer(onehot_node)
        res_value = graph.node['one_hot_d']['value']
        assert np.array_equal(exp_value, int64_array(res_value))
