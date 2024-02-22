# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.ConcatOptimization import ConcatOdInputEraserAndPortsReconnect
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, shaped_const_with_data, connect


class ConcatOdInputEraserAndPortsReconnectTest(unittest.TestCase):
    def test_deletion(self):
        nodes = {
            **shaped_const_with_data('input_0', [1]),
            **shaped_const_with_data('input_1', [1]),
            **shaped_const_with_data('input_2', [0]),
            **shaped_const_with_data('input_3', [1]),
            **regular_op_with_shaped_data('concat', [3], {'type': 'Concat'}),
            **result(),
        }
        edges_before = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]
        edges_after = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_3', '2:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges_before, nodes_with_edges_only=True)
        ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges_after, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_deletion_2(self):
        nodes = {
            **shaped_const_with_data('input_0', [5, 0]),
            **shaped_const_with_data('input_1', [5, 1]),
            **shaped_const_with_data('input_2', [5, 3]),
            **shaped_const_with_data('input_3', [5, 5]),
            **regular_op_with_shaped_data('concat', [5, 9], {'type': 'Concat', 'axis': 1}),
            **result(),
        }
        edges_before = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]
        edges_after = [
            *connect('input_1', '0:concat'),
            *connect('input_2', '1:concat'),
            *connect('input_3', '2:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges_before, nodes_with_edges_only=True)
        ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges_after, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_deletion_3(self):
        nodes = {
            **shaped_const_with_data('input_0', [5, 3]),
            **shaped_const_with_data('input_1', [5, 1]),
            **shaped_const_with_data('input_2', [5, 5]),
            **shaped_const_with_data('input_3', [5, 0]),
            **regular_op_with_shaped_data('concat', [5, 9], {'type': 'Concat', 'axis': 1}),
            **result(),
        }
        edges_before = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]
        edges_after = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges_before, nodes_with_edges_only=True)
        ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges_after, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_deletion_unconnected_port_and_0d(self):
        nodes = {
            **shaped_const_with_data('input_0', [5, 3]),
            **shaped_const_with_data('input_2', [5, 1]),
            **shaped_const_with_data('input_3', [5, 0]),
            **regular_op_with_shaped_data('concat', [5, 4], {'type': 'Concat', 'axis': 1}),
            **result(),
        }
        edges_before = [
            *connect('input_0', '0:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]
        edges_after = [
            *connect('input_0', '0:concat'),
            *connect('input_2', '1:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges_before, nodes_with_edges_only=True)
        ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges_after, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_deletion_unconnected_ports(self):
        nodes = {
            **shaped_const_with_data('input_0', [5, 3]),
            **shaped_const_with_data('input_4', [5, 1]),
            **shaped_const_with_data('input_7', [5, 2]),
            **regular_op_with_shaped_data('concat', [5, 6], {'type': 'Concat', 'axis': 1}),
            **result(),
        }
        edges_before = [
            *connect('input_0', '0:concat'),
            *connect('input_4', '4:concat'),
            *connect('input_7', '7:concat'),
            *connect('concat', 'output'),
        ]
        edges_after = [
            *connect('input_0', '0:concat'),
            *connect('input_4', '1:concat'),
            *connect('input_7', '2:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges_before, nodes_with_edges_only=True)
        ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges_after, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_deletion_trailing_unconnected_ports(self):
        nodes = {
            **shaped_const_with_data('input_0', [5, 3]),
            **regular_op_with_shaped_data('concat', [5, 3], {'type': 'Concat', 'axis': 1}),
            **result(),
        }
        edges_before = [
            *connect('input_0', '0:concat'),
            *connect('concat', 'output'),
        ]
        edges_after = [
            *connect('input_0', '0:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges_before, nodes_with_edges_only=True)
        Node(graph, 'concat').add_input_port(1)
        ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges_after, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(1 not in Node(graph, 'concat').in_ports())

    def test_negative(self):
        nodes = {
            **shaped_const_with_data('input_0', [1]),
            **shaped_const_with_data('input_1', [1]),
            **shaped_const_with_data('input_2', [1]),
            **shaped_const_with_data('input_3', [1]),
            **regular_op_with_shaped_data('concat', [4], {'type': 'Concat'}),
            **result(),
        }
        edges = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_assertion_error(self):
        nodes = {
            **shaped_const_with_data('input_0', [0]),
            **shaped_const_with_data('input_1', [0]),
            **shaped_const_with_data('input_2', [0]),
            **shaped_const_with_data('input_3', [0]),
            **regular_op_with_shaped_data('concat', [0], {'type': 'Concat'}),
            **result(),
        }
        edges = [
            *connect('input_0', '0:concat'),
            *connect('input_1', '1:concat'),
            *connect('input_2', '2:concat'),
            *connect('input_3', '3:concat'),
            *connect('concat', 'output'),
        ]

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        self.assertRaises(AssertionError, ConcatOdInputEraserAndPortsReconnect().find_and_replace_pattern, graph)
