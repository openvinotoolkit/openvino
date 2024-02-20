# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.FuseReshapesSequence import FuseReshapesSequenceKaldi
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, valued_const_with_data, connect, regular_op_with_shaped_data


class FuseReshapesKaldiTests(unittest.TestCase):
    ref_nodes = {
        **regular_op_with_shaped_data('conv', [1, 128, 1, 9], {'kind': 'op', 'op': 'Convolution',
                                                               'kernel': [1, 11, 1, 5], 'patch_stride': 5,
                                                               'kernel_spatial': [1, 5]}),
        **valued_const_with_data('transpose_out_order', int64_array([0, 2, 3, 1])),
        **regular_op_with_shaped_data('transpose_out', [1, 1, 9, 128], {'op': 'Transpose', 'type': 'Transpose'}),
        **valued_const_with_data('transpose_in_order', int64_array([0, 3, 1, 2])),
        **regular_op_with_shaped_data('transpose_in', [1, 128, 1, 9], {'op': 'Transpose', 'type': 'Transpose'}),
        **regular_op_with_shaped_data('pool', [1, 128, 1, 3], {'kind': 'op', 'op': 'Pooling',
                                                               'pool_stride': 3, 'pool_step': [1, 1, 1, 1]}),
    }

    nodes = {
        **regular_op_with_shaped_data('conv', [1, 128, 1, 9], {'kind': 'op', 'op': 'Convolution',
                                                               'kernel': [1, 1, 11, 5]}),
        **valued_const_with_data('transpose_out_order', int64_array([0, 2, 3, 1])),
        **regular_op_with_shaped_data('transpose_out', [1, 1, 9, 128], {'op': 'Transpose', 'type': 'Transpose'}),
        **valued_const_with_data('reshape_out_shape', int64_array([0, -1])),
        **regular_op_with_shaped_data('reshape_out', [1, 1152], {'op': 'Reshape', 'type': 'Reshape',
                                                                 'special_zero': True}),

        **regular_op_with_shaped_data('shapeof', [4], {'op': 'ShapeOf', 'type': 'ShapeOf'}),
        **valued_const_with_data('ind', int64_array([0])),
        **valued_const_with_data('axis', int64_array(0)),
        **regular_op_with_shaped_data('gather_batch', [], {'op': 'Gather', 'type': 'Gather'}),
        **valued_const_with_data('t', int64_array([1])),
        **valued_const_with_data('h', int64_array([9])),
        **valued_const_with_data('ind_h', int64_array([1])),
        **regular_op_with_shaped_data('gather_h', [], {'op': "Gather", 'type': 'Gather'}),
        **valued_const_with_data('th', int64_array([9])),
        **regular_op_with_shaped_data('div', [], {'op': 'Div', 'type': 'Divide'}),
        **regular_op_with_shaped_data('concat', [4], {'op': 'Concat', 'type': 'Concat'}),

        **regular_op_with_shaped_data('reshape_in', [1, 1, 9, 128], {'op': 'Reshape', 'type': 'Reshape'}),
        **valued_const_with_data('transpose_in_order', int64_array([0, 3, 1, 2])),
        **regular_op_with_shaped_data('transpose_in', [1, 128, 1, 9], {'op': 'Transpose', 'type': 'Transpose'}),
        **regular_op_with_shaped_data('pool', [1, 128, 1, 3], {'kind': 'op', 'op': 'Pooling', 'pool_stride': 3,
                                                               'pool_step': [1, 1, 1, 1]}),
    }

    def test_conv_reshape_pool(self):
        graph = build_graph(self.nodes, [
            *connect('conv', '0:transpose_out'),
            *connect('transpose_out_order', '1:transpose_out'),
            *connect('transpose_out', '0:reshape_out'),
            *connect('reshape_out_shape', '1:reshape_out'),
            *connect('reshape_out', 'shapeof'),

            *connect('shapeof', '0:gather_batch'),
            *connect('ind', '1:gather_batch'),
            *connect('axis', '2:gather_batch'),
            *connect('shapeof', '0:gather_h', skip_data=True),
            *connect('ind_h', '1:gather_h'),
            *connect('axis', '2:gather_h', skip_data=True),
            *connect('gather_h', '0:div'),
            *connect('th', '1:div'),
            *connect('gather_batch', '0:concat'),
            *connect('t', '1:concat'),
            *connect('h', '2:concat'),
            *connect('div', '3:concat'),
            *connect('concat', '1:reshape_in'),

            *connect('reshape_out', '0:reshape_in', skip_data=True),
            *connect('reshape_in', '0:transpose_in'),
            *connect('transpose_in_order', "1:transpose_in"),
            *connect('transpose_in', 'pool'),
        ], nodes_with_edges_only=True)

        FuseReshapesSequenceKaldi().find_and_replace_pattern(graph)

        ref_graph = build_graph(self.ref_nodes,
                                [
                                    *connect('conv', '0:transpose_out'),
                                    *connect('transpose_out_order', '1:transpose_out'),
                                    *connect('transpose_out', '0:transpose_in'),
                                    *connect('transpose_in_order', "1:transpose_in"),
                                    *connect('transpose_in', 'pool'),
                                ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'pool')
        self.assertTrue(flag, resp)
