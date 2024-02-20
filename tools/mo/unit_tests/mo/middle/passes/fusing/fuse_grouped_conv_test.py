# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from openvino.tools.mo.ops.op import PermuteAttrs
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, regular_op_with_shaped_data, regular_op, shaped_data, \
    valued_const_with_data, shaped_const_with_data, valued_data

nodes = {
    **regular_op_with_shaped_data('placeholder1', [1, 16, 10, 10], {'type': 'Parameter'}),

    **valued_const_with_data('split_1_axis', int64_array(1), {'type': 'Const'}),
    **regular_op('split_1', {'type': 'Split', 'can_be_fused': True}),
    **shaped_data('split_1_data1', [1, 4, 10, 10]),
    **shaped_data('split_1_data2', [1, 4, 10, 10]),
    **shaped_data('split_1_data3', [1, 4, 10, 10]),
    **shaped_data('split_1_data4', [1, 4, 10, 10]),

    **shaped_const_with_data('split_2_in_const_weights', int64_array([3, 3, 4, 16]), {'type': 'Const'}),
    **regular_op('split_2', {'type': 'Split'}),
    **valued_data('split_2_data1', np.zeros([3, 3, 4, 4])),
    **valued_data('split_2_data2', np.zeros([3, 3, 4, 4])),
    **valued_data('split_2_data3', np.zeros([3, 3, 4, 4])),
    **valued_data('split_2_data4', np.zeros([3, 3, 4, 4])),

    **regular_op_with_shaped_data('conv2d_1', [1, 4, 8, 8],
                                  {'type': 'Convolution', 'channel_dims': np.array([1]), 'pad': np.array([2, 2]),
                                   'stride': np.array([2, 2]),
                                   'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([3, 2, 0, 1]),
                                                                                   inv=int64_array([2, 3, 1, 0])),
                                   'group': 1, 'output': 4, 'output_shape': [1, 4, 8, 8], 'can_be_fused': True}),
    **regular_op_with_shaped_data('conv2d_2', [1, 4, 8, 8],
                                  {'type': 'Convolution', 'pad': np.array([2, 2]), 'stride': np.array([2, 2]),
                                   'can_be_fused': True}),
    **regular_op_with_shaped_data('conv2d_3', [1, 4, 8, 8],
                                  {'type': 'Convolution', 'pad': np.array([2, 2]), 'stride': np.array([2, 2]),
                                   'can_be_fused': True}),
    **regular_op_with_shaped_data('conv2d_4', [1, 4, 8, 8],
                                  {'type': 'Convolution', 'pad': np.array([2, 2]), 'stride': np.array([2, 2]),
                                   'can_be_fused': True}),

    **regular_op_with_shaped_data('concat', [1, 16, 8, 8], {'type': 'Concat', 'axis': np.array(1)}),

    **regular_op_with_shaped_data('fused_group_conv', [1, 16, 8, 8],
                                  {'type': 'Convolution', 'channel_dims': np.array([1]), 'pad': np.array([2, 2]),
                                   'stride': np.array([2, 2]),
                                   'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([3, 2, 0, 1]),
                                                                                   inv=int64_array([2, 3, 1, 0])),
                                   'group': 1, 'output': 4, 'output_shape': [1, 4, 8, 8], 'can_be_fused': True}),
    **shaped_const_with_data('new_weights_const', int64_array([3, 3, 4, 16]), {'type': 'Const'}),

    **result('result')
}


class FuseGroupedConvTest(unittest.TestCase):
    def test_fuse_grouped_conv(self):
        graph = build_graph(nodes, [*connect('placeholder1', '0:split_1'), *connect('split_1_axis', '1:split_1'),
                                    ('split_1', 'split_1_data1', {'out': 0}),
                                    ('split_1', 'split_1_data2', {'out': 1}),
                                    ('split_1', 'split_1_data3', {'out': 2}),
                                    ('split_1', 'split_1_data4', {'out': 3}),

                                    *connect('split_2_in_const_weights', 'split_2'),
                                    ('split_2', 'split_2_data1', {'out': 0}),
                                    ('split_2', 'split_2_data2', {'out': 1}),
                                    ('split_2', 'split_2_data3', {'out': 2}),
                                    ('split_2', 'split_2_data4', {'out': 3}),

                                    ('split_1_data1', 'conv2d_1', {'in': 0}),
                                    ('split_1_data2', 'conv2d_2', {'in': 0}),
                                    ('split_1_data3', 'conv2d_3', {'in': 0}),
                                    ('split_1_data4', 'conv2d_4', {'in': 0}),

                                    ('split_2_data1', 'conv2d_1', {'in': 1}),
                                    ('split_2_data2', 'conv2d_2', {'in': 1}),
                                    ('split_2_data3', 'conv2d_3', {'in': 1}),
                                    ('split_2_data4', 'conv2d_4', {'in': 1}),

                                    *connect('conv2d_1', '0:concat'),
                                    *connect('conv2d_2', '1:concat'),
                                    *connect('conv2d_3', '2:concat'),
                                    *connect('conv2d_4', '3:concat'),

                                    *connect('concat', 'result')])

        graph_ref = build_graph(nodes, [*connect('placeholder1', '0:fused_group_conv'),
                                        *connect('new_weights_const', '1:fused_group_conv'),
                                        *connect('fused_group_conv', 'result')])

        graph.graph['layout'] = 'NCHW'
        grouped_convolutions_fusing(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

        group_conv_node = Node(graph, 'conv2d_1')
        group_conv_weights_shape = group_conv_node.in_node(1).shape
        self.assertTrue((group_conv_weights_shape == int64_array([3, 3, 4, 16])).all())
