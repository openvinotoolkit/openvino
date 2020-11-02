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

from extensions.ops.cumsum import CumSum
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, valued_const_with_data, regular_op_with_shaped_data, result, connect

nodes_attributes = {
    **regular_op_with_shaped_data('data', [1, 3, 224, 224], {'type': 'Parameter', 'value': None,
                                                             '_out_port_data_type': {0: np.float32}}),
    **valued_const_with_data('axis', int64_array(0)),
    **regular_op_with_shaped_data('cumsum', None, {'op': 'CumSum', 'type': 'CumSum', 'name': 'cumsum'}),
    **regular_op_with_shaped_data('identity', None, {'op': 'Identity', 'name': 'identity'}),
    **result('output'),
}


class TestCumSum(unittest.TestCase):
    def test_cumsum_axis(self):
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:cumsum'),
                             *connect('axis', '1:cumsum'),
                             *connect('cumsum', '0:identity'),
                             ('identity', 'identity_d', {'out': 0}),
                             ('identity_d', 'output'),
                             ],
                            {'cumsum': {'reverse': False, 'exclusive': False}
                             }, nodes_with_edges_only=True)

        cumsum_node = Node(graph, 'cumsum')
        CumSum.infer(cumsum_node)
        self.assertTrue(np.array_equal(cumsum_node.out_port(0).data.get_shape(), int64_array([1, 3, 224, 224])))

    def test_cumsum_value_prop(self):
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:cumsum'),
                             *connect('axis', '1:cumsum'),
                             ('cumsum', 'cumsum_d', {'out': 0}),
                             ('cumsum_d', 'output'),
                             ],
                            {'data_d': {'value': np.array([1., 2., 3., 4., 5.]).astype(np.float32), 'shape': [5]},
                             'cumsum': {'reverse': False, 'exclusive': False}
                             }, nodes_with_edges_only=True)

        cumsum_node = Node(graph, 'cumsum')
        CumSum.infer(cumsum_node)
        self.assertTrue(np.array_equal(cumsum_node.out_port(0).data.get_value(),
                                       np.array([1., 3., 6., 10., 15.]).astype(np.float32)))

    def test_cumsum_value_prop_exclusive(self):
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:cumsum'),
                             *connect('axis', '1:cumsum'),
                             ('cumsum', 'cumsum_d', {'out': 0}),
                             ('cumsum_d', 'output'),
                             ],
                            {'data_d': {'value': np.array([1., 2., 3., 4., 5.]).astype(np.float32), 'shape': [5]},
                             'cumsum': {'reverse': False, 'exclusive': True}
                             }, nodes_with_edges_only=True)

        cumsum_node = Node(graph, 'cumsum')
        CumSum.infer(cumsum_node)
        self.assertTrue(np.array_equal(cumsum_node.out_port(0).data.get_value(),
                                       np.array([0., 1., 3., 6., 10.]).astype(np.float32)))

    def test_cumsum_value_prop_reverse(self):
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:cumsum'),
                             *connect('axis', '1:cumsum'),
                             ('cumsum', 'cumsum_d', {'out': 0}),
                             ('cumsum_d', 'output'),
                             ],
                            {'data_d': {'value': np.array([1., 2., 3., 4., 5.]).astype(np.float32), 'shape': [5]},
                             'cumsum': {'reverse': True, 'exclusive': False}
                             }, nodes_with_edges_only=True)

        cumsum_node = Node(graph, 'cumsum')
        CumSum.infer(cumsum_node)
        self.assertTrue(np.array_equal(cumsum_node.out_port(0).data.get_value(),
                                       np.array([15., 14., 12., 9., 5.]).astype(np.float32)))

    def test_cumsum_value_prop_exclusive_reverse(self):
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:cumsum'),
                             *connect('axis', '1:cumsum'),
                             ('cumsum', 'cumsum_d', {'out': 0}),
                             ('cumsum_d', 'output'),
                             ],
                            {'data_d': {'value': np.array([1., 2., 3., 4., 5.]).astype(np.float32), 'shape': [5]},
                             'cumsum': {'reverse': True, 'exclusive': True}
                             }, nodes_with_edges_only=True)

        cumsum_node = Node(graph, 'cumsum')
        CumSum.infer(cumsum_node)
        self.assertTrue(np.array_equal(cumsum_node.out_port(0).data.get_value(),
                                       np.array([14., 12., 9., 5., 0.]).astype(np.float32)))

    def test_cumsum_value_prop_axis_1(self):
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:cumsum'),
                             *connect('axis', '1:cumsum'),
                             ('cumsum', 'cumsum_d', {'out': 0}),
                             ('cumsum_d', 'output'),
                             ],
                            {'data_d': {'value': np.array([[1., 2., 3.], [4., 5., 6.]]).astype(np.float32),
                                        'shape': [2, 3]},
                             'axis_d': {'value': int64_array(1),
                                        'shape': []},
                             'cumsum': {'reverse': False, 'exclusive': False}
                             }, nodes_with_edges_only=True)

        cumsum_node = Node(graph, 'cumsum')
        CumSum.infer(cumsum_node)
        self.assertTrue(np.array_equal(cumsum_node.out_port(0).data.get_value(),
                                       np.array([[1., 3., 6.], [4., 9., 15.]]).astype(np.float32)))
