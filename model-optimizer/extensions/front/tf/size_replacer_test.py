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

from extensions.front.tf.SizeReplacer import SizeFrontReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, connect, \
    valued_const_with_data

nodes = lambda output_type: {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('size', {'op': 'Size', 'type': None, 'output_type': output_type, 'name': 'my_size'}),
    **result(),

    **regular_op_with_empty_data('shape', {'type': 'ShapeOf', 'output_type': output_type}),
    **valued_const_with_data('zero', int64_array([0])),
    **regular_op_with_empty_data('reduce', {'type': 'ReduceProd', 'keep_dims': False}),
}


@generator
class SizeReplacerTest(unittest.TestCase):

    @generate(np.int32, np.int64)
    def test_size_replacer(self, output_type):
        graph = build_graph(nodes_attrs=nodes(output_type), edges=[
            *connect('input', 'size'),
            *connect('size', 'output'),
        ], nodes_with_edges_only=True)
        SizeFrontReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes(output_type), edges=[
            *connect('input', 'shape'),
            *connect('shape', '0:reduce'),
            *connect('zero', '1:reduce'),
            *connect('reduce', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertEqual(graph.get_op_nodes(type='ReduceProd')[0]['name'], 'my_size',
                         'Name is not inherited from original node for SizeReplacer')
        print(output_type)

    def test_size_replacer_assertion(self):
        graph = build_graph(nodes_attrs=nodes(None), edges=[
            *connect('input', 'size'),
            *connect('size', 'output'),
        ], nodes_with_edges_only=True)
        self.assertRaises(AssertionError, SizeFrontReplacer().find_and_replace_pattern, graph)
