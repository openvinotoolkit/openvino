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

from extensions.middle.FakeSplitOutputs import AddFakeOutputsToSplit, AddFakeOutputsToVariadicSplit
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Node
from mo.middle.passes.eliminate import graph_clean_up
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter', 'shape': np.array([1, 227, 227, 3])},
    # VariadicSplit operation
    'variadic_split': {'type': 'VariadicSplit', 'kind': 'op', 'op': 'VariadicSplit'},
    'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 3, 'axis': 3},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None, 'infer': copy_shape_infer},
    'res': {'type': 'Result', 'kind': 'op', 'op': 'Result'},
    # Data nodes
    'placeholder_data': {'kind': 'data', 'value': None, 'shape': np.array([1, 227, 227, 3])},
    'variadic_split_data_1': {'kind': 'data', 'value': None, 'shape': np.array([1, 2, 227, 3])},
    'split_data_1': {'kind': 'data', 'value': None, 'shape': np.array([1, 227, 227, 1])},
    'last_data': {'kind': 'data', 'value': None, 'shape': np.array([1, 227, 227, 3])},

    'axis_const': {'kind': 'op'},
    'axis_const_data': {'value': np.int64(1), 'shape': None, 'kind': 'data'},
    'split_dim_const': {'kind': 'op'},
    'split_dim_const_data': {'value': np.array([1, 2, 3]), 'shape': None, 'kind': 'data'},

}


class SplitSaveEmptyBranchesTest(unittest.TestCase):
    def test_variadic_split_non_zero(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_data'), ('placeholder_data', 'variadic_split'),
                             ('variadic_split', 'variadic_split_data_1'), ('variadic_split_data_1', 'last'),
                             ('last', 'last_data'), ('last_data', 'res'),

                             ('axis_const', 'axis_const_data'),
                             ('split_dim_const', 'split_dim_const_data'),
                             ('axis_const_data', 'variadic_split', {'in': 1}),
                             ('split_dim_const_data', 'variadic_split', {'in': 2}),
                             ], nodes_with_edges_only=True)
        node = Node(graph, 'variadic_split')

        # extractor should do it
        node['out_ports_count'] = 3
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        replacer = AddFakeOutputsToVariadicSplit()
        replacer.find_and_replace_pattern(graph)

        for n in graph.get_op_nodes():
            n['need_shape_inference'] = False
        graph_clean_up(graph)

        self.assertTrue(len(node.out_edges()) == 3)

    def test_split(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_data'), ('placeholder_data', 'split'),
                             ('split', 'split_data_1'), ('split_data_1', 'last'),
                             ('last', 'last_data'), ('last_data', 'res'),
                             ], nodes_with_edges_only=True)
        node = Node(graph, 'split')

        # extractor should do it
        node['out_ports_count'] = node.num_splits
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        replacer = AddFakeOutputsToSplit()
        replacer.find_and_replace_pattern(graph)

        for n in graph.get_op_nodes():
            n['need_shape_inference'] = False
        graph_clean_up(graph)

        self.assertTrue(len(node.out_edges()) == node.num_splits)
