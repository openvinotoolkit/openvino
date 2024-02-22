# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.back.TransposeDFT import TransposeDFT
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect, \
    regular_op_with_empty_data

dft_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', [8, 2, 40, 56], {'type': 'Parameter', 'op': 'Parameter'}),
    **valued_const_with_data('axes', int64_array([-2, -1])),
    **regular_op_with_shaped_data('dft', [8, 2, 40, 56], {'op': 'DFT', 'need_insert_transposes_for_dft': True}),
    **regular_op_with_shaped_data('abs', [8, 2, 40, 56], {'type': 'Abs', 'op': 'Abs'}),
    **result(),
}

dft_graph_edges = [
    *connect('placeholder', '0:dft'),
    *connect('axes', '1:dft'),
    *connect('dft', 'abs'),
    *connect('abs', 'output'),
]


transposed_dft_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', [8, 2, 40, 56], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_empty_data('transpose_before',
                                 {'type': 'Transpose', 'op': 'Transpose', 'need_shape_inference': True}),
    **valued_const_with_data('transpose_before_axis_const', int64_array([0, 2, 3, 1])),
    **regular_op_with_empty_data('transpose_after',
                                 {'type': 'Transpose', 'op': 'Transpose', 'need_shape_inference': True}),
    **valued_const_with_data('transpose_after_axis_const', int64_array([0, 3, 1, 2])),
    **valued_const_with_data('dft_axes', int64_array([-2, -1])),
    **regular_op_with_shaped_data('dft', [8, 2, 40, 56], {'op': 'DFT', 'need_insert_transposes_for_dft': True}),
    **regular_op_with_shaped_data('abs', [8, 2, 40, 56], {'type': 'Abs', 'op': 'Abs'}),
    **result(),
}

transposed_dft_graph_edges = [
    *connect('placeholder', '0:transpose_before'),
    *connect('transpose_before_axis_const', '1:transpose_before'),
    *connect('transpose_before', '0:dft'),
    *connect('dft_axes', '1:dft'),
    *connect('dft', '0:transpose_after'),
    *connect('transpose_after_axis_const', '1:transpose_after'),
    *connect('transpose_after', 'abs'),
    *connect('abs', 'output'),
]


nontransposed_dft_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', [8, 2, 40, 56], {'type': 'Parameter', 'op': 'Parameter'}),
    **valued_const_with_data('axes', int64_array([-2, -1])),
    **regular_op_with_shaped_data('dft', [8, 2, 40, 56], {'op': 'DFT'}),
    **regular_op_with_shaped_data('abs', [8, 2, 40, 56], {'type': 'Abs', 'op': 'Abs'}),
    **result(),
}

nontransposed_dft_graph_edges = [
    *connect('placeholder', '0:dft'),
    *connect('axes', '1:dft'),
    *connect('dft', 'abs'),
    *connect('abs', 'output'),
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
