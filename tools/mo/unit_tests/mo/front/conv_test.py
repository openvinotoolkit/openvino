# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer
from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.middle.passes.conv import fuse_pad
from openvino.tools.mo.middle.passes.fusing.mark_unfused_nodes import mark_shape_of_sugraph_as_unfusable
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.ops.elementwise import eltwise_infer
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.pad import Pad
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, \
    shaped_parameter, valued_const_with_data


class PadFusingTest(unittest.TestCase):

    # standard case: not in the shape subgraph
    def test_pad_fusing(self):
        nodes = {
            **shaped_parameter('input', shape_array([1, 3, 248, 248])),

            **valued_const_with_data('pads_begin', shape_array([0, 0, 1, 1])),
            **valued_const_with_data('pads_end', shape_array([0, 0, 1, 1])),
            **valued_const_with_data('fill_value', shape_array(0.0)),
            **valued_const_with_data('weights', shape_array(np.zeros([3, 16, 4, 4]))),

            **regular_op_with_empty_data('pad', {'type': 'Pad',
                                                 'op': 'Pad',
                                                 'infer': Pad.infer,
                                                 'mode': 'constant'}),

            **regular_op_with_empty_data('conv', {'type': 'Convolution',
                                                  'op': 'Convolution',
                                                  'infer': Convolution.infer,
                                                  # zeros, no paddings
                                                  'pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                                                  'dilation': np.array([1, 1, 1, 1]),
                                                  'stride': np.array([1, 1, 1, 1]),
                                                  'group': 1,
                                                  'kernel_spatial_idx': np.array([2, 3]),
                                                  'output': 64,
                                                  'spatial_dims': np.array([2, 3]),
                                                  'channel_dims': np.array([1]),
                                                  'batch_dims': np.array([0]),
                                                  'input_feature_channel': 1,
                                                  'output_feature_channel': 0}),
            **result(),
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('input', '0:pad'),
            *connect('pads_begin', '1:pad'),
            *connect('pads_end', '2:pad'),
            *connect('fill_value', '3:pad'),
            *connect('pad', '0:conv'),
            *connect('weights', '1:conv'),
            *connect('conv', 'output'),
        ], nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'middle'

        graph = partial_infer(graph)
        mark_shape_of_sugraph_as_unfusable(graph)
        for_graph_and_each_sub_graph_recursively(graph, fuse_pad)
        graph.clean_up()

        conv_fused_with_pad = regular_op_with_empty_data('conv', {'type': 'Convolution',
                                                                  'op': 'Convolution',
                                                                  # ones are taken from fused Pad
                                                                  'pad': np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
                                                                  'dilation': np.array([1, 1, 1, 1]),
                                                                  'stride': np.array([1, 1, 1, 1]),
                                                                  'group': 1,
                                                                  'kernel_spatial_idx': np.array([2, 3]),
                                                                  'output': 64,
                                                                  'spatial_dims': np.array([2, 3]),
                                                                  'channel_dims': np.array([1]),
                                                                  'batch_dims': np.array([0]),
                                                                  'input_feature_channel': 1,
                                                                  'output_feature_channel': 0,

                                                                  'infer': Convolution.infer})

        graph_ref = build_graph(nodes_attrs=nodes, update_attributes=conv_fused_with_pad, edges=[
            *connect('input', '0:conv'),
            *connect('weights', '1:conv'),
            *connect('conv', 'output'),
        ], nodes_with_edges_only=True)
        graph_ref.graph['layout'] = 'NCHW'
        graph_ref.stage = 'middle'

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Pad in the shape subgraph
    def test_pad_fusing_shape_subgraph(self):
        nodes = {
            **shaped_parameter('input', shape_array([1, 3, 1020, 1020])),
            **regular_op_with_empty_data('input_shape', {'type': 'ShapeOf', 'op': 'ShapeOf', 'output_type': np.int64,
                                                         'infer': Shape.infer}),
            **regular_op_with_empty_data('gathered_shape', {'type': 'Gather', 'batch_dims': 0, 'infer': Gather.infer}),
            **valued_const_with_data('axis', np.array([0])),
            **valued_const_with_data('indices', np.array([2, 3])),

            **regular_op_with_empty_data('div', {'type': 'Div',
                                                 'infer': lambda node: eltwise_infer(node, lambda a, b: a / b)}),
            **regular_op_with_empty_data('sub_1', {'type': 'Sub',
                                                   'infer': lambda node: eltwise_infer(node, lambda a, b: a - b)}),
            **regular_op_with_empty_data('sub_2', {'type': 'Sub',
                                                   'infer': lambda node: eltwise_infer(node, lambda a, b: a - b)}),

            **valued_const_with_data('div_const', shape_array([2])),
            **valued_const_with_data('sub_const', shape_array([512])),

            **regular_op_with_empty_data('pad', {'type': 'Pad',
                                                 'op': 'Pad',
                                                 'infer': Pad.infer,
                                                 'mode': 'constant'}),

            **regular_op_with_empty_data('concat', {'type': 'Concat',
                                                    'op': 'Concat',
                                                    'axis': 0,
                                                    'infer': concat_infer}),

            **valued_const_with_data('pad_end', shape_array([0, 0, 0, 0])),
            **valued_const_with_data('blank_zeros', shape_array([0, 0])),

            **regular_op_with_empty_data('conv', {'type': 'Convolution',
                                                  'op': 'Convolution',

                                                  'pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                                                  'dilation': np.array([1, 1, 1, 1]),
                                                  'stride': np.array([1, 1, 1, 1]),
                                                  'group': 1,
                                                  'kernel_spatial_idx': np.array([2, 3]),
                                                  'output': 64,
                                                  'spatial_dims': np.array([2, 3]),
                                                  'channel_dims': np.array([1]),
                                                  'batch_dims': np.array([0]),
                                                  'input_feature_channel': 1,
                                                  'output_feature_channel': 0,

                                                  'infer': Convolution.infer}),
            **valued_const_with_data('weights', shape_array(np.zeros([3, 16, 4, 4]))),
            **result(),
        }

        graph = build_graph(nodes_attrs=nodes,
                            update_attributes={
                                'gathered_shape_d': {'kind': 'data', 'value': shape_array([256, 256]),
                                                     'shape': shape_array([2])}},
                            edges=[
                                *connect('input', 'input_shape', skip_data=True),
                                *connect('input_shape', '0:gathered_shape'),
                                *connect('indices', '1:gathered_shape'),
                                *connect('axis', '2:gathered_shape'),

                                *connect('gathered_shape', 'sub_1'),
                                *connect('sub_const', 'sub_1'),
                                *connect('sub_1', 'div'),
                                *connect('div_const', 'div'),
                                *connect('div', '0:sub_2'),
                                *connect('sub_1', '1:sub_2'),
                                *connect('input', '0:pad'),

                                *connect('blank_zeros', '0:concat'),
                                *connect('sub_2', '1:concat'),
                                *connect('concat', '1:pad'),

                                *connect('pad_end', '2:pad'),
                                *connect('pad', '0:conv'),
                                *connect('weights', '1:conv'),
                                *connect('conv', 'output'),
                            ], nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'middle'

        graph = partial_infer(graph)

        # graph must remain unchanged
        graph_ref = graph.copy()

        mark_shape_of_sugraph_as_unfusable(graph)
        for_graph_and_each_sub_graph_recursively(graph, fuse_pad)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
