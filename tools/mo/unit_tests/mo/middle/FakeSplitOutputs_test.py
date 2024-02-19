# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.FakeSplitOutputs import AddFakeOutputsToSplit, AddFakeOutputsToVariadicSplit
from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.eliminate import graph_clean_up
from unit_tests.utils.graph import build_graph

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

    'axis_const': {'kind': 'op', 'op': 'Const'},
    'axis_const_data': {'value': np.int64(1), 'shape': None, 'kind': 'data'},
    'split_dim_const': {'kind': 'op', 'op': 'Const'},
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
