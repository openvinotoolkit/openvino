"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.front.tf.WhereDecomposition import WhereDecomposition
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': None,
        'kind': 'data',
        'data_type': None
    },
    'tf_where': {'op': 'Where', 'kind': 'op'},
    'tf_where_data': {'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}


graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'tf_where'),
    ('tf_where', 'tf_where_data'),
    ('tf_where_data', 'output'),
]


ref_graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': None,
        'kind': 'data',
        'data_type': None
    },
    'non_zero': {'kind': 'op', 'op': 'NonZero', 'output_type': np.int64},
    'non_zero_data': {'kind': 'data'},
    'transpose': {'kind': 'op', 'op': 'Transpose'},
    'transpose_data': {'kind': 'data'},
    'perm_const': {'kind': 'op', 'op': 'Const', 'shape': [2], 'value': int64_array([1, 0])},
    'perm_const_data': {'kind': 'data', 'shape': [2], 'value': int64_array([1, 0])},
    'output': {'kind': 'op', 'op': 'Result'},
}

ref_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('placeholder_data', 'non_zero'),
    ('non_zero', 'non_zero_data'),
    ('non_zero_data', 'transpose', {'in': 0}),
    ('perm_const', 'perm_const_data'),
    ('perm_const_data', 'transpose', {'in': 1}),
    ('transpose', 'transpose_data'),
    ('transpose_data', 'output'),
]


@generator
class TFWhereDecompositionTest(unittest.TestCase):
    @generate(*[[1, 100, 120, 150], [16, 125, 14]])
    def test_1(self, input_shape):
        in_shape = int64_array(input_shape)
        graph = build_graph(graph_node_attrs,
                            graph_edges,
                            update_attributes={
                                'placeholder_data': {'shape': in_shape}
                            })
        WhereDecomposition().find_and_replace_pattern(graph)
        ref_graph = build_graph(ref_graph_node_attrs,
                                ref_graph_edges,
                                update_attributes={
                                    'placeholder_data': {'shape': in_shape}
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
