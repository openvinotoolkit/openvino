# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.StridedSliceReplacer import ReplaceStridedSliceWithSqueezeUnsqueeze
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import regular_op_with_shaped_data, regular_op_with_empty_data, shaped_const_with_data, \
    result, connect, build_graph

nodes = {
    **regular_op_with_shaped_data('input', [1, 3, 5, 5], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_empty_data('strided_slice', {'type': 'StridedSlice', 'op': 'StridedSlice',
                                                   'begin_mask': [0, 0, 0, 0], 'end_mask': [0, 0, 0, 0]}),
    **shaped_const_with_data('begin', [4]),
    **shaped_const_with_data('end', [4]),
    **result('result'),

    **regular_op_with_empty_data('squeeze', {'type': 'Squeeze', 'op': 'Squeeze'}),
    **shaped_const_with_data('squeeze_axes', None),

    **regular_op_with_empty_data('unsqueeze', {'type': 'Unsqueeze', 'op': 'Unsqueeze'}),
    **shaped_const_with_data('unsqueeze_axes', None)
}

pattern_edges = [
    *connect('input', '0:strided_slice'),
    *connect('begin', '1:strided_slice'),
    *connect('end', '2:strided_slice'),
    *connect('strided_slice', 'result')
]


class TestStridedSliceReplacer(unittest.TestCase):

    def test_negative_different_input_and_output_shapes(self):
        graph = build_graph(
            nodes_attrs=nodes,
            edges=pattern_edges,
            update_attributes={
                'strided_slice_d': {'shape': [1, 3, 3, 3]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = graph.copy()

        ReplaceStridedSliceWithSqueezeUnsqueeze().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_replace_with_squeeze(self):
        graph = build_graph(
            nodes_attrs=nodes,
            edges=pattern_edges,
            update_attributes={
                'strided_slice': {'shrink_axis_mask': [1, 0, 0, 0], 'new_axis_mask': [0, 0, 0, 0]},
                'strided_slice_d': {'shape': [3, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes,
            edges=[
                *connect('input', '0:squeeze'),
                *connect('squeeze_axes', '1:squeeze'),
                *connect('squeeze', 'result')
            ],
            update_attributes={
                'squeeze_axes_d': {'value': [0]},
                'squeeze_d': {'shape': [3, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ReplaceStridedSliceWithSqueezeUnsqueeze().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_replace_with_unsqueeze(self):
        graph = build_graph(
            nodes_attrs=nodes,
            edges=pattern_edges,
            update_attributes={
                'strided_slice': {'shrink_axis_mask': [0, 0, 0, 0], 'new_axis_mask': [1, 0, 0, 0]},
                'strided_slice_d': {'shape': [1, 1, 3, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes,
            edges=[
                *connect('input', '0:unsqueeze'),
                *connect('unsqueeze_axes', '1:unsqueeze'),
                *connect('unsqueeze', 'result')
            ],
            update_attributes={
                'unsqueeze_axes_d': {'value': [0]},
                'unsqueeze_d': {'shape': [1, 1, 3, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ReplaceStridedSliceWithSqueezeUnsqueeze().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
