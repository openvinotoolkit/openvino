# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np

from openvino.tools.mo.middle.GatherNDDecomposition import GatherNDDecomposition
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


nodes = {
    'input': {'kind': 'op', 'op': 'Const'},
    'input_data': {'kind': 'data'},

    'indices_input': {'kind': 'op', 'op': 'Const'},
    'indices_input_data': {'kind': 'data'},

    'gathernd': {'kind': 'op', 'op': 'GatherND'},
    'gathernd_data': {'kind': 'data'},

    'result': {'kind': 'op', 'op': 'Result'},
}

edges = [
    ('input', 'input_data'),
    ('input_data', 'gathernd', {'in': 0}),

    ('indices_input', 'indices_input_data'),
    ('indices_input_data', 'gathernd', {'in': 1}),

    ('gathernd', 'gathernd_data'),
    ('gathernd_data', 'result'),
]

nodes_expected = {
    'input': {'kind': 'op', 'op': 'Const'},
    'input_data': {'kind': 'data'},

    'reshape_shape': {'kind': 'op', 'op': 'Const'},
    'reshape_shape_data': {'kind': 'data'},

    'reshape': {'kind': 'op', 'op': 'Reshape'},
    'reshape_data': {'kind': 'data'},

    'axis': {'kind': 'op', 'op': 'Const'},
    'axis_data': {'kind': 'data'},

    'indices': {'kind': 'op', 'op': 'Const'},
    'indices_data': {'kind': 'data'},

    'gather': {'kind': 'op', 'op': 'Gather'},
    'gather_data': {'kind': 'data'},

    'result': {'kind': 'op', 'op': 'Result'},
}

edges_expected = [
    ('input', 'input_data'),
    ('input_data', 'reshape', {'in': 0}),

    ('reshape_shape', 'reshape_shape_data'),
    ('reshape_shape_data', 'reshape', {'in': 1}),

    ('reshape', 'reshape_data'),
    ('reshape_data', 'gather', {'in': 0}),

    ('indices', 'indices_data'),
    ('indices_data', 'gather', {'in': 1}),

    ('axis', 'axis_data'),
    ('axis_data', 'gather', {'in': 2}),

    ('gather', 'gather_data'),
    ('gather_data', 'result'),
]


class GatherNDDecompositionTest(unittest.TestCase):

    def test_GatherNDDecomposition_2by2indices_validinputs(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 2]).reshape([2, 1])},
                                'indices_input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 0, 0, 0]).reshape([2, 2])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 2]).reshape([2, 1])},
                                    'indices': {'shape': np.array([2]), 'value': np.array([1, 0])},
                                    'reshape_shape': {'shape': np.array([1]), 'value': np.array([-1])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_2by2indices_invalidinputs(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                'indices_input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 0, 0, 0]).reshape([2, 2])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = graph

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_2by1indices_validinputs(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                'indices_input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 0]).reshape([2, 1])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                    'indices': {'shape': np.array([2]), 'value': np.array([1, 0])},
                                    'reshape_shape': {'shape': np.array([2]), 'value': np.array([-1, 2])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_2by0indices_invalidinputs(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([1, 2]), 'value': np.array([1, 2]).reshape([1, 2])},
                                'indices_input_data': {'shape': np.array([2]), 'value': np.array([1, 0])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = graph

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_2by0indices_validinputs(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 2])},
                                'indices_input_data': {'shape': np.array([2]), 'value': np.array([1, 0])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 2])},
                                    'indices': {'shape': np.array([]), 'value': np.array([1])},
                                    'reshape_shape': {'shape': np.array([1]), 'value': np.array([-1])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_1leadingdim(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                'indices_input_data': {'shape': np.array([2, 1, 1]), 'value': np.array([1, 0]).reshape([2, 1, 1])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                    'indices': {'shape': np.array([2, 1]), 'value': np.array([1, 0]).reshape([2, 1])},
                                    'reshape_shape': {'shape': np.array([2]), 'value': np.array([-1, 2])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_3leadingdims(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                'indices_input_data': {'shape': np.array([2, 1, 1, 1, 1]), 'value': np.array([1, 0]).reshape([2, 1, 1, 1, 1])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                    'indices': {'shape': np.array([2, 1, 1, 1]), 'value': np.array([1, 0]).reshape([2, 1, 1, 1])},
                                    'reshape_shape': {'shape': np.array([2]), 'value': np.array([-1, 2])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_nonzerobatchdim(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                'indices_input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 0]).reshape([2, 1])},
                                'gathernd': {'batch_dims': 1}
                            },
                            nodes_with_edges_only=True)
        graph_ref = graph

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_complexexample1_nonzerobatchdim(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 3, 4]), 'value': np.array([i for i in range(24)]).reshape([2, 3, 4])},
                                'indices_input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 0]).reshape([2, 1])},
                                'gathernd': {'batch_dims': 1}
                            },
                            nodes_with_edges_only=True)
        graph_ref = graph

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_complexexample2(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 3, 4]), 'value': np.array([i for i in range(24)]).reshape([2, 3, 4])},
                                'indices_input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 0]).reshape([2, 1])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([2, 3, 4]), 'value': np.array([i for i in range(24)]).reshape([2, 3, 4])},
                                    'indices': {'shape': np.array([2]), 'value': np.array([1, 0]).reshape([2])},
                                    'reshape_shape': {'shape': np.array([3]), 'value': np.array([-1, 3, 4])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_complexexample3(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([1, 1, 5]), 'value': np.array([1, 2, 3, 4, 5]).reshape([1, 1, 5])},
                                'indices_input_data': {'shape': np.array([2, 2, 3]), 'value': np.array([0, 0, 3, 0, 0, 1, 0, 0, 4, 0, 0, 2]).reshape([2, 2, 3])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([1, 1, 5]), 'value': np.array([1, 2, 3, 4, 5]).reshape([1, 1, 5])},
                                    'indices': {'shape': np.array([2, 2]), 'value': np.array([3, 1, 4, 2]).reshape([2, 2])},
                                    'reshape_shape': {'shape': np.array([1]), 'value': np.array([-1])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_complexexample4(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([1, 4, 1]), 'value': np.array([1, 2, 3, 4]).reshape([1, 4, 1])},
                                'indices_input_data': {'shape': np.array([2, 2, 2]), 'value': np.array([0, 1, 0, 3, 0, 2, 0, 0]).reshape([2, 2, 2])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([1, 4, 1]), 'value': np.array([1, 2, 3, 4]).reshape([1, 4, 1])},
                                    'indices': {'shape': np.array([2, 2]), 'value': np.array([1, 3, 2, 0]).reshape([2, 2])},
                                    'reshape_shape': {'shape': np.array([2]), 'value': np.array([-1, 1])},
                                    'axis': {'shape': np.array([]), 'value': 0}
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_dynamic_data_shape(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([1, -1, 1]), 'value': np.array([1, 2, 3, 4]).reshape([1, 4, 1])},
                                'indices_input_data': {'shape': np.array([2, 2, 2]), 'value': np.array([0, 1, 0, 3, 0, 2, 0, 0]).reshape([2, 2, 2])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = graph

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_dynamic_indices_shape(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([1, 4, 1]), 'value': np.array([1, 2, 3, 4]).reshape([1, 4, 1])},
                                'indices_input_data': {'shape': np.array([2, -1, 2]), 'value': np.array([0, 1, 0, 3, 0, 2, 0, 0]).reshape([2, 2, 2])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)
        graph_ref = graph

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
