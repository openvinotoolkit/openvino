# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.back.SpecialNodesFinalization import CreateConstNodesReplacement
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs


class CreateConstNodesReplacementTest(unittest.TestCase):
    nodes = [
        ('data_node', {'kind': 'data', 'shape': None, 'value': None}),
        ('next_node', {'kind': 'op'}),
    ]
    edges = [
        ('data_node', 'next_node')
    ]

    new_nodes = [
        ('const', {'kind': 'op', 'op': 'Const'}),
        ('const_data', {'kind': 'data'})
    ]
    new_edges = [
        ('const', 'data_node'),
        ('const_data', 'const')
    ]

    def test_one_node(self):
        """We should add Const node and data node."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})]
        )
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + self.new_nodes,
            edges_with_attrs=self.edges + self.new_edges,
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data}),
                                     ('const_data', {'shape': shape, 'value': data})]
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='next_node')
        self.assertTrue(flag, resp)

    def test_one_bin_node(self):
        """Nothing should happen."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})],
            update_edge_attrs={('data_node', 'next_node', 0): {'bin': 0}},
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph, last_node='next_node')
        self.assertTrue(flag, resp)

    def test_two_nodes_with_bin(self):
        """Test case for data node with 2 consumers with bin edge attr.
        Nothing should happened."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + [('next_node_2', {'kind': 'op'})],
            edges_with_attrs=self.edges + [('data_node', 'next_node_2')],
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})],
            update_edge_attrs={('data_node', 'next_node', 0): {'bin': 0}, ('data_node', 'next_node_2', 0): {'bin': 0}},
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph, last_node='next_node')
        self.assertTrue(flag, resp)

    def test_two_nodes_one_bin(self):
        """Test case for two output nodes, one with 'bin' parameter, other without."""
        shape = np.array([2, 3, 4])
        data = np.zeros(shape)
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + [('next_node_2', {'kind': 'op'})],
            edges_with_attrs=self.edges + [('data_node', 'next_node_2')],
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data})],
            update_edge_attrs={('data_node', 'next_node', 0): {'bin': 0}},
        )
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + self.new_nodes + [('next_node_2', {'kind': 'op'})],
            edges_with_attrs=self.edges + self.new_edges + [('data_node', 'next_node_2')],
            update_nodes_attributes=[('data_node', {'shape': shape, 'value': data}),
                                     ('const_data', {'shape': shape, 'value': data})]
        )
        tested_pattern = CreateConstNodesReplacement()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='next_node')
        self.assertTrue(flag, resp)
