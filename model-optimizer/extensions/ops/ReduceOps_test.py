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
from generator import generate, generator

from extensions.ops.ReduceOps import reduce_infer
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, result, connect, valued_const_with_data

nodes_attributes = {
    **regular_op_with_shaped_data('data', [1, 3, 224, 224], {'type': 'Parameter', 'value': None,
                                                             '_out_port_data_type': {0: np.float32}}),
    **valued_const_with_data('axis', int64_array(0)),
    **regular_op_with_shaped_data('reduce_lp', None, {'op': 'ReduceLp', 'type': None, 'name': 'my_reduce_lp'}),
    **regular_op_with_shaped_data('identity', None, {'op': 'Identity', 'name': 'identity'}),
    **result('output'),
}


@generator
class ReduceLpTest(unittest.TestCase):
    @generate(*[
        ([3, 2, 2], [0], True, 1),
        ([3, 2, 2], [0], True, 2),
        ([3, 2, 2], [1], True, 2),
        ([3, 2, 2], [2], True, 2),
        ([3, 2, 2], [0], False, 1),
        ([3, 2, 2], [0], False, 2),
        ([3, 2, 2], [1], False, 2),
        ([3, 2, 2], [2], False, 2),
    ])
    def test_reduce_lp(self, shape, axes, keepdims, p):
        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        reduced = np.power(np.sum(a=np.abs(np.power(data, p)), axis=tuple(axes), keepdims=keepdims), 1 / p)
        axis = int64_array(axes)
        p = int64_array(p)
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:reduce_lp'),
                             *connect('axis', '1:reduce_lp'),
                             *connect('reduce_lp', '0:identity'),
                             ('identity', 'identity_d', {'out': 0}),
                             ('identity_d', 'output')
                             ],
                            {'data_d': {'value': data, 'shape': data.shape},
                             'axis_d': {'value': axis, 'shape': axis.shape},
                             'reduce_lp': {'keep_dims': keepdims}},
                            nodes_with_edges_only=True)

        reduce_node = Node(graph, 'reduce_lp')
        reduce_node.op = reduce_node.type = 'ReduceL' + str(p)
        reduce_infer(reduce_node)
        self.assertTrue(np.array_equal(reduce_node.out_port(0).data.get_value(), reduced))
