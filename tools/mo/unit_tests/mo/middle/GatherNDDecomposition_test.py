# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from openvino.tools.mo.middle.GatherNDDecomposition import GatherNDDecomposition
from generator import generator, generate
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, result, connect, connect_data, \
    valued_const_with_data, valued_data, const
from openvino.tools.mo.front.common.partial_infer.utils import int64_array

""" original_data_shape = np.array([2, 1])
original_indices = np.array([[1, 0], [0, 0]])
batch_dims = np.array(0)
axis = np.array(0)

expected_reshape_shape = np.array([-1])
#expected_reshape_shape = np.array([2, 1])
expected_new_indices = np.array([1, 0]) """

""" nodes_ref = {
    # inputs
    **regular_op_with_shaped_data('data', original_data_shape,
                                  {'type': 'Parameter',
                                   'op': 'Parameter',
                                   'shape': original_data_shape,
                                   'data_type': np.int64}),

    # constants
    # **const('indices', original_indices, original_indices.shape),
    **valued_const_with_data('indices', original_indices),

    # operators
    **regular_op_with_shaped_data('gathernd', original_data_shape, {'type': 'GatherND', 'op': 'GatherND', 'batch_dims': 0}),

    # result node
    **result('result'),
}

nodes_expected = {
    # inputs
    **regular_op_with_shaped_data('data', original_data_shape,
                                  {'type': 'Parameter',
                                   'op': 'Parameter',
                                   'shape': original_data_shape,
                                   'data_type': np.int64}),

    # expected
    **valued_const_with_data('indices', expected_new_indices),
    **valued_const_with_data('reshape_shape', expected_reshape_shape),
    **valued_const_with_data('axis', axis),
    # **const('indices', expected_new_indices, expected_new_indices.shape),
    # **const('reshape_shape', expected_reshape_shape, expected_reshape_shape.shape),

    # operators
    **regular_op_with_shaped_data('reshape', original_data_shape, {'type': 'Reshape', 'op': 'Reshape'}),
    **regular_op_with_shaped_data('gather', expected_reshape_shape, {'type': 'Gather', 'op': 'Gather'}),

    # result node
    **result('result'),
} """

nodes = {

    'input': {'kind': 'op', 'op': 'Const'},
    'input_data': {'kind': 'data'},

    'indices_input': {'kind': 'op', 'op': 'Const'},
    'indices_input_data': {'kind': 'data'},

    'gathernd': {'kind': 'op', 'op': 'GatherND'},
    'gathernd_data': {'kind': 'data', 'shape': None, 'value': None},

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

    def GatherNDDecomposition_2by2indices_validinputs(self):
        """ graph = build_graph(nodes_ref, [
            *connect('data', '0:gathernd'),
            *connect('indices', '1:gathernd'),
            *connect('gathernd', 'result'),
        ],
            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_expected, [
            *connect('data', '0:reshape'),
            *connect('reshape_shape', '1:reshape'),
            *connect('reshape', '0:gather'),
            *connect('indices', '1:gather'),
            *connect('axis', '2:gather'),
            *connect('gather', 'result'),
        ],
            nodes_with_edges_only=True) """

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
                                    'axis': {'shape': np.array([]), 'value': 0},
                                },
                                nodes_with_edges_only=True)

        nx.draw_networkx(graph_ref, font_size=3)
        plt.savefig('ref.svg')
        plt.clf()

        nx.draw_networkx(graph, font_size=3)
        plt.savefig('exp.svg')
        plt.clf()

        for node in graph.node:
            print(f'\nNode: {node}')

        GatherNDDecomposition().find_and_replace_pattern(graph)
        #import pudb
        # pu.db
        # graph.clean_up()

        nx.draw_networkx(graph, font_size=3)
        plt.savefig('after.svg')
        plt.clf()

        for node in graph.node:
            print(f'\nNode: {node}')

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GatherNDDecomposition_2by1indices_validinputs(self):

        graph = build_graph(nodes,
                            edges,
                            update_attributes={
                                'input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                'indices_input_data': {'shape': np.array([2, 2]), 'value': np.array([1, 0]).reshape([2, 1])},
                                'gathernd': {'batch_dims': 0}
                            },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_expected,
                                edges_expected,
                                update_attributes={
                                    'input_data': {'shape': np.array([2, 1]), 'value': np.array([1, 2, 3, 4]).reshape([2, 2])},
                                    'indices': {'shape': np.array([2]), 'value': np.array([1, 0])},
                                    'reshape_shape': {'shape': np.array([2]), 'value': np.array([-1, 1])},
                                    'axis': {'shape': np.array([]), 'value': 0},
                                },
                                nodes_with_edges_only=True)

        GatherNDDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(
            graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
