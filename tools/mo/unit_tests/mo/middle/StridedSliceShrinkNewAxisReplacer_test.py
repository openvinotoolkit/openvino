# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.middle.StridedSliceShrinkNewAxisReplacer import \
    ReplaceStridedSliceShrinkNewAxisWithSqueezeUnsqueeze
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.strided_slice import StridedSlice
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import regular_op_with_empty_data, shaped_const_with_data, \
    result, connect, build_graph, shaped_parameter, valued_const_with_data

nodes = {
    **shaped_parameter('input', [1, 3, 5, 5]),
    **regular_op_with_empty_data('strided_slice', {'type': 'StridedSlice', 'op': 'StridedSlice',
                                                   'infer': StridedSlice.infer,
                                                   'begin_mask': [0, 0, 0, 0], 'end_mask': [0, 0, 0, 0],
                                                   'ellipsis_mask': [0, 0, 0, 0]}),
    **valued_const_with_data('begin', shape_array([0, 0, 0, 0])),
    **valued_const_with_data('end', shape_array([1, 1, 1, 1])),
    **result('result'),

    **regular_op_with_empty_data('squeeze', {'type': 'Squeeze', 'op': 'Squeeze', 'infer': Squeeze.infer}),
    **shaped_const_with_data('squeeze_axes', None),

    **regular_op_with_empty_data('unsqueeze', {'type': 'Unsqueeze', 'op': 'Unsqueeze', 'infer': Unsqueeze.infer}),
    **shaped_const_with_data('unsqueeze_axes', None)
}

pattern_edges = [
    *connect('input', '0:strided_slice'),
    *connect('begin', '1:strided_slice'),
    *connect('end', '2:strided_slice'),
    *connect('strided_slice', 'result')
]


class TestStridedSliceShrinkNewAxisReplacer(unittest.TestCase):

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
                *connect('input', '0:strided_slice'),
                *connect('begin', '1:strided_slice'),
                *connect('end', '2:strided_slice'),

                *connect('strided_slice', '0:squeeze'),
                *connect('squeeze_axes', '1:squeeze'),
                *connect('squeeze', 'result'),
            ],

            update_attributes={
                'squeeze_axes_d': {'value': [0]},
                'squeeze_d': {'shape': [3, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ReplaceStridedSliceShrinkNewAxisWithSqueezeUnsqueeze().find_and_replace_pattern(graph)
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
                *connect('input', '0:strided_slice'),
                *connect('begin', '1:strided_slice'),
                *connect('end', '2:strided_slice'),
                *connect('strided_slice', '0:unsqueeze'),
                *connect('unsqueeze_axes', '1:unsqueeze'),
                *connect('unsqueeze', 'result')
            ],
            update_attributes={
                'unsqueeze_axes_d': {'value': [0]},
                'unsqueeze_d': {'shape': [1, 1, 3, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ReplaceStridedSliceShrinkNewAxisWithSqueezeUnsqueeze().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_replace_shrink_new_axis_with_squeeze_unsqueeze(self):
        graph = build_graph(
            nodes_attrs=nodes,
            edges=pattern_edges,
            update_attributes={
                # input_shape = (1, 3, 5, 5)
                'strided_slice': {'shrink_axis_mask': [1, 0, 0, 0], 'new_axis_mask': [0, 0, 1, 1]},
                'strided_slice_d': {'shape': [3, 1, 1, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = build_graph(
            nodes_attrs=nodes,
            edges=[
                *connect('input', '0:strided_slice'),
                *connect('begin', '1:strided_slice'),
                *connect('end', '2:strided_slice'),
                *connect('strided_slice', '0:squeeze'),
                *connect('squeeze_axes', '1:squeeze'),
                *connect('squeeze', '0:unsqueeze'),
                *connect('unsqueeze_axes', '1:unsqueeze'),
                *connect('unsqueeze', 'result')
            ],
            update_attributes={
                'squeeze_axes_d': {'value': [0]},
                'squeeze_d': {'shape': [3, 5, 5]},
                'unsqueeze_axes_d': {'value': [1, 2]},
                'unsqueeze_d': {'shape': [3, 1, 1, 5, 5]}
            },
            nodes_with_edges_only=True
        )

        ReplaceStridedSliceShrinkNewAxisWithSqueezeUnsqueeze().find_and_replace_pattern(graph)
        graph = partial_infer(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_no_shrink_new_axis_masks(self):
        graph = build_graph(
            nodes_attrs=nodes,
            edges=pattern_edges,
            update_attributes={
                'strided_slice_d': {'shape': [1, 3, 3, 3]}
            },
            nodes_with_edges_only=True
        )

        ref_graph = graph.copy()

        ReplaceStridedSliceShrinkNewAxisWithSqueezeUnsqueeze().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
