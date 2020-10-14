"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

import numpy as np
from generator import generator, generate

from extensions.ops.one_hot import OneHot
from mo.front.common.partial_infer.utils import int64_array, float_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, connect


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


@generator
class TestOneHotInfer(unittest.TestCase):
    @generate(*[
        # 0d input
        (1, [0, 1, 0, 0]),
        # 1d input
        ([1, 2], [[0, 1, 0, 0], [0, 0, 1, 0]]),
        # 2D input
        ([[1, 2], [3, 4]], [[[0, 1, 0, 0], [0, 0, 1, 0]],
                            [[0, 0, 0, 1], [0, 0, 0, 0]]]),
        # 3d input
        ([[[0, 2], [1, 2]], [[2, 1], [3, 0]]],
         [[[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [0, 0, 1, 0]]],
          [[[0, 0, 1, 0], [0, 1, 0, 0]], [[0, 0, 0, 1], [1, 0, 0, 0]]]]),
        # 1d input with negative indices
        ([-2, 2], [[0, 0, 1, 0], [0, 0, 1, 0]]),
        # check if axis is neither 0 nor -1
        ([[1, 2], [3, 4]], [[[0, 0], [1, 0], [0, 1], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [1, 0]]], 1)
    ])
    def test_infer(self, input_value, exp_value, axis=-1):
        graph = build_graph(generate_nodes(int64_array(input_value), axis), edges)
        onehot_node = Node(graph, 'one_hot')
        OneHot.infer(onehot_node)
        res_value = graph.node['one_hot_d']['value']
        self.assertTrue(np.array_equal(exp_value, int64_array(res_value)))
