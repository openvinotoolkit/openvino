# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.back.TransposeDFT import TransposeDFT
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


dft_graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([8, 2, 40, 56]),
        'kind': 'data',
        'data_type': None
    },
    'axes': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([-2, -1]),
        'shape': int64_array([2]),
    },
    'axes_data': {
        'kind': 'data',
        'value': int64_array([-2, -1]),
        'shape': int64_array([2]),
    },
    'dft': {'kind': 'op', 'op': 'DFT', 'need_insert_transposes_for_dft': True},
    'dft_data': {
        'kind': 'data',
        'shape': int64_array([8, 2, 40, 56]),
        'value': None,
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([8, 2, 40, 56]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

dft_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('axes', 'axes_data'),
    ('placeholder_data', 'dft', {'in': 0}),
    ('axes_data', 'dft', {'in': 1}),
    ('dft', 'dft_data'),
    ('dft_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


transposed_dft_graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([8, 2, 40, 56]),
        'kind': 'data',
        'data_type': None
    },

    'transpose_before': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'need_shape_inference': True},
    'transpose_before_data': {'value': None, 'shape': None, 'kind': 'data'},
    'transpose_before_axis_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([0, 2, 3, 1])},
    'transpose_before_axis_const_data': {'kind': 'data', 'value': int64_array([0, 2, 3, 1]), 'shape': None},
    'transpose_after': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'need_shape_inference': True},
    'transpose_after_data': {'value': None, 'shape': None, 'kind': 'data'},
    'transpose_after_axis_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([0, 3, 1, 2])},
    'transpose_after_axis_const_data': {'kind': 'data', 'value': int64_array([0, 3, 1, 2])},

    'dft_axes': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([-2, -1]),
        'shape': int64_array([2]),
    },
    'dft_axes_data': {
        'kind': 'data',
        'value': int64_array([-2, -1]),
        'shape': int64_array([2]),
    },
    'dft': {'kind': 'op', 'op': 'DFT', 'need_insert_transposes_for_dft': True},
    'dft_data': {
        'kind': 'data',
        'shape': int64_array([8, 2, 40, 56]),
        'value': None,
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([8, 2, 40, 56]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

transposed_dft_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('dft_axes', 'dft_axes_data'),
    ('transpose_before_axis_const', 'transpose_before_axis_const_data'),
    ('transpose_after_axis_const', 'transpose_after_axis_const_data'),
    ('transpose_before', 'transpose_before_data'),
    ('transpose_after', 'transpose_after_data'),
    ('transpose_before_axis_const_data', 'transpose_before', {'in': 1}),
    ('transpose_after_axis_const_data', 'transpose_after', {'in': 1}),
    ('placeholder_data', 'transpose_before', {'in': 0}),
    ('transpose_before_data', 'dft', {'in': 0}),
    ('dft_axes_data', 'dft', {'in': 1}),
    ('dft', 'dft_data'),
    ('dft_data', 'transpose_after', {'in': 0}),
    ('transpose_after_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


nontransposed_dft_graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([8, 40, 56, 2]),
        'kind': 'data',
        'data_type': None
    },
    'axes': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([-2, -1]),
        'shape': int64_array([2]),
    },
    'axes_data': {
        'kind': 'data',
        'value': int64_array([-2, -1]),
        'shape': int64_array([2]),
    },
    'dft': {'kind': 'op', 'op': 'DFT'},
    'dft_data': {
        'kind': 'data',
        'shape': int64_array([8, 40, 56, 2]),
        'value': None,
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([8, 40, 56, 2]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

nontransposed_dft_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('axes', 'axes_data'),
    ('placeholder_data', 'dft', {'in': 0}),
    ('axes_data', 'dft', {'in': 1}),
    ('dft', 'dft_data'),
    ('dft_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


class TransposeDFTTest(unittest.TestCase):
    def test_dft_transpose(self):
        graph = build_graph(nodes_attrs=dft_graph_node_attrs, edges=dft_graph_edges)
        ref_graph = build_graph(nodes_attrs=transposed_dft_graph_node_attrs, edges=transposed_dft_graph_edges)
        graph.graph['fw'] = 'tf'
        TransposeDFT().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_dft_nontranspose(self):
        graph = build_graph(nodes_attrs=nontransposed_dft_graph_node_attrs, edges=nontransposed_dft_graph_edges)
        ref_graph = build_graph(nodes_attrs=nontransposed_dft_graph_node_attrs, edges=nontransposed_dft_graph_edges)
        TransposeDFT().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
