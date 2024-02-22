# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.tile import Tile, AttributedTile
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'op': {'kind': 'op'},
    'data': {'value': None, 'shape': np.array([10, 20, 30, 40]), 'kind': 'data'},
    'const': {'kind': 'op'},
    'tile_values': {'value': None, 'shape': np.array([4]), 'kind': 'data'},
    'tile': {'type': 'AttributedTile', 'kind': 'op'},
    'tile_out': {'value': None, 'shape': None, 'kind': 'data'},

}

edges = [
    ('op', 'data'),
    ('data', 'tile'),
    ('const', 'tile_values'),
    ('tile_values', 'tile'),
    ('tile', 'tile_out'),
]

attributed_edges = [
    ('op', 'data'),
    ('data', 'tile'),
    ('tile', 'tile_out'),
]


class TestTileInfer(unittest.TestCase):
    def test_tile_infer_correct(self):
        graph = build_graph(nodes_attributes, edges,
                            {'tile_values': {'value': np.array([7, 1, 1, 1])}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.array([70, 20, 30, 40]) == graph.node['tile_out']['shape']))

    def test_tile_infer_correct_2(self):
        graph = build_graph(nodes_attributes, edges,
                            {'tile_values': {'value': np.array([1, 7, 1, 1])}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.array([10, 140, 30, 40]) == graph.node['tile_out']['shape']))

    def test_tile_infer_correct_2d_tensor(self):
        graph = build_graph(nodes_attributes, edges,
                            {'data': {'shape': np.array([3, 7])},
                             'tile_values': {'value': np.array([5, 1])}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.array([15, 7]) == graph.node['tile_out']['shape']))

    def test_tile_infer_all_ones(self):
        graph = build_graph(nodes_attributes, edges,
                            {'tile_values': {'value': np.array([1, 1, 1, 1])}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.array([10, 20, 30, 40]) == graph.node['tile_out']['shape']))

    def test_tile_infer_two_non_one(self):
        graph = build_graph(nodes_attributes, edges,
                            {'tile_values': {'value': np.array([2, 1, 1, 2])}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.array([20, 20, 30, 80]) == graph.node['tile_out']['shape']))

    def test_tile_infer_three_non_one(self):
        graph = build_graph(nodes_attributes, edges,
                            {'tile_values': {'value': np.array([2, 1, 5, 2])}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.array([20, 20, 150, 80]) == graph.node['tile_out']['shape']))

    def test_tile_infer_none_input_shape(self):
        graph = build_graph(nodes_attributes, edges,
                            {'data': {'shape': None},
                             'tile_values': {'value': np.array([1, 7, 1, 1])}})
        tile_node = Node(graph, 'tile')
        self.assertRaises(AssertionError, Tile.infer, tile_node)

    def test_tile_infer_values_test(self):
        input_data = np.arange(-30, 60, 0.25).reshape([2, 4, 3, -1])
        tile_values = np.array([3, 1, 1, 1])
        graph = build_graph(nodes_attributes, edges,
                            {'data': {'shape': np.array(input_data.shape), 'value': input_data},
                             'tile_values': {'value': tile_values}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.tile(input_data, tile_values) == graph.node['tile_out']['value']))

    def test_tile_infer_values_const_propagation(self):
        """
        Test for constant propagation even if tile with multiple tile indices is not supported
        """
        input_data = np.arange(-30, 60, 0.25).reshape([2, 4, 3, -1])
        tile_values = np.array([4, 3, 2, 5])
        graph = build_graph(nodes_attributes, edges,
                            {'data': {'shape': np.array(input_data.shape), 'value': input_data},
                             'tile_values': {'value': tile_values}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.tile(input_data, tile_values) == graph.node['tile_out']['value']))

    def test_tile_infer_undefined_tile_values(self):
        graph = build_graph(nodes_attributes, edges,
                            {'tile_values': {'value': None}})
        tile_node = Node(graph, 'tile')
        self.assertRaises(AssertionError, Tile.infer, tile_node)

    def test_tile_infer_shapes_alignment(self):
        graph = build_graph(nodes_attributes, edges,
                            {'tile_values': {'value': np.array([1, 2, 3]), 'shape': np.array([3])}})
        tile_node = Node(graph, 'tile')
        Tile.infer(tile_node)
        self.assertTrue(np.all(np.array([10, 20, 60, 120]) == graph.node['tile_out']['shape']))

    def test_tile_infer_one_input_correct(self):
        graph = build_graph(nodes_attributes, attributed_edges,
                            {'tile': {'axis': 1, 'tiles': 7}})
        tile_node = Node(graph, 'tile')
        AttributedTile.infer(tile_node)
        self.assertTrue(np.all(np.array([10, 140, 30, 40]) == graph.node['tile_out']['shape']))
        self.assertEqual(tile_node.axis, 1)
        self.assertEqual(tile_node.tiles, 7)

    def test_tile_infer_one_input_correct_missing_axis(self):
        graph = build_graph(nodes_attributes, attributed_edges,
                            {'tile': {'tiles': 7}})
        tile_node = Node(graph, 'tile')
        self.assertRaises(AssertionError, AttributedTile.infer, tile_node)

    def test_tile_infer_one_input_correct_missing_tiles(self):
        graph = build_graph(nodes_attributes, attributed_edges,
                            {'tile': {'axis': 1}})
        tile_node = Node(graph, 'tile')
        self.assertRaises(AssertionError, AttributedTile.infer, tile_node)
