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

from extensions.middle.L2NormToNorm import L2NormToNorm
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph_with_attrs

shape = (1, 300, 300, 3)
weights_value = np.array([1.0, 1.0, 1.0])

# A list with nodes attributes used to build various graphs.
nodes = [
    ('input', dict(kind='op', shape=shape, op='Parameter', data_type=np.float32)),
    ('input_data', dict(kind='data', shape=shape, data_type=np.float32)),
    ('l2_normalize', dict(kind='op', op='Mul', name='l2_norm_name')),
    ('l2_normalize_data', dict(kind='data')),
    ('maximum', dict(kind='op', op='Maximum')),
    ('maximum_data', dict(kind='data')),
    ('maximum_y_const', dict(kind='op', op='Const', value=np.array(12.e-13, dtype=np.float32))),
    ('maximum_y_data', dict(kind='data', value=np.array(12.e-13, dtype=np.float32))),
    ('rsqrt_pow', dict(kind='data', value=-0.5)),
    ('rsqrt', dict(kind='op', op='Pow')),
    ('rsqrt_data', dict(kind='data')),
    ('square_pow', dict(kind='op', op='Const', value=2.)),
    ('square_pow_data', dict(kind='data', value=2.)),
    ('square', dict(kind='op', op='Pow')),
    ('square_data', dict(kind='data')),
    ('sum', dict(kind='op', op='ReduceSum')),
    ('sum_data', dict(kind='data')),
    # nodes added after replacement
    ('normalize_node', dict(kind='op', op='Normalize')),
    ('weights_node', dict(kind='op', op='Const', shape=weights_value.shape, value=weights_value)),
    ('weights_node_data', dict(kind='data', op='Const')),
    ('result', dict(kind='op', op='Result'))
]

edges = [
    ('input', 'input_data', {'out': 0}),
    ('input_data', 'square', {'in': 0}),
    ('square_pow', 'square_pow_data', {'out': 0}),
    ('square_pow_data', 'square', {'in': 1}),
    ('square', 'square_data'),
    ('square_data', 'sum'),
    ('sum', 'sum_data'),
    ('maximum_y_const', 'maximum_y_data'),
    ('maximum_y_data', 'maximum'),
    ('sum_data', 'maximum'),
    ('maximum', 'maximum_data'),
    ('maximum_data', 'rsqrt', {'in': 0}),
    ('rsqrt_pow', 'rsqrt', {'in': 1}),
    ('rsqrt', 'rsqrt_data'),
    ('rsqrt_data', 'l2_normalize'),
    ('input_data', 'l2_normalize'),
    ('l2_normalize', 'l2_normalize_data'),
    ('l2_normalize_data', 'result'),
]

edges_after_replacement = [
    ('input', 'input_data', {'out': 0}),
    ('input_data', 'normalize_node'),
    ('weights_node', 'weights_node_data'),
    ('weights_node_data', 'normalize_node'),
    ('normalize_node', 'l2_normalize_data'),
    ('l2_normalize_data', 'result'),
]


class L2NormToNormTest(unittest.TestCase):
    def test_single_consumer(self):
        graph = build_graph_with_attrs(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'middle'
        L2NormToNorm().find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(nodes, edges_after_replacement, nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(type='Normalize')[0]]['name'] == 'l2_norm_name')
        self.assertTrue(flag, resp)

    def test_multiple_consumers(self):
        graph = build_graph_with_attrs(nodes + [('result_2', dict(kind='op', op='Result'))],
                                       edges + [('input_data', 'result_2')], nodes_with_edges_only=True)
        graph.stage = 'middle'

        L2NormToNorm().find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(nodes + [('result_2', dict(kind='op', op='Result'))],
                                           edges_after_replacement+ [('input_data', 'result_2')],
                                           nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)

        self.assertTrue(graph.node[graph.get_nodes_with_attributes(type='Normalize')[0]]['name'] == 'l2_norm_name')
        self.assertTrue(flag, resp)
