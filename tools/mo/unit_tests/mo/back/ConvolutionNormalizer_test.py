# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.back.ConvolutionNormalizer import PullReshapeThroughFQ, V7ConvolutionWithGroupsResolver, \
    V10ConvolutionWithGroupsResolver
from openvino.tools.mo.back.ShapeOfConstFolding import ShapeOfConstFolding
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.fakequantize import FakeQuantize
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, regular_op_with_empty_data, \
    valued_const_with_data, connect


def graph_template(weights_initial_shape, new_reshape_shape, limits_initial_shape, limits_new_shape=None):
    limits_new_shape = limits_initial_shape if limits_new_shape is None else limits_new_shape

    core_connections = [
        *connect('input:0', '0:convolution'),
        *connect('convolution:0', '0:output'),
    ]

    core_nodes = lambda weights_shape, limit_shape, reshape_shape: {
        **regular_op_with_shaped_data('input', None, {'type': 'Parameter', 'op': 'Parameter'}),

        **valued_const_with_data('weights', np.ones(weights_shape)),

        **valued_const_with_data('dim', int64_array(reshape_shape)),
        **regular_op_with_shaped_data('reshape', reshape_shape, {'type': 'Reshape', 'infer': Reshape.infer, 'op': 'Reshape'}),

        **valued_const_with_data('il', np.ones(limit_shape)),
        **valued_const_with_data('ih', np.ones(limit_shape)),
        **valued_const_with_data('ol', np.ones(limit_shape)),
        **valued_const_with_data('oh', np.ones(limit_shape)),

        **regular_op_with_shaped_data('FQ', weights_shape, {'type': 'FakeQuantize', 'infer': FakeQuantize.infer,
                                                            'stop_value_propagation': True, 'levels': 2, 'op': 'FakeQuantize'}),

        **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution', 'op': 'Convolution'}),

        **result(),
    }

    nodes_before = core_nodes(weights_initial_shape, limits_initial_shape, new_reshape_shape)
    edges_before = [

        *connect('weights:0', '0:FQ'),
        *connect('il:0', '1:FQ'),
        *connect('ih:0', '2:FQ'),
        *connect('ol:0', '3:FQ'),
        *connect('oh:0', '4:FQ'),

        *connect('FQ:0', '0:reshape'),
        *connect('dim:0', '1:reshape'),
        *connect('reshape:0', '1:convolution'),

        *core_connections,
    ]
    graph = build_graph(nodes_attrs=nodes_before, edges=edges_before, nodes_with_edges_only=True)

    nodes_after = core_nodes(new_reshape_shape, limits_new_shape, [])
    edges_after = [
        *connect('weights:0', '0:FQ'),
        *connect('il:0', '1:FQ'),
        *connect('ih:0', '2:FQ'),
        *connect('ol:0', '3:FQ'),
        *connect('oh:0', '4:FQ'),
        *connect('FQ:0', '1:convolution'),

        *core_connections,
    ]
    graph_ref = build_graph(nodes_attrs=nodes_after, edges=edges_after, nodes_with_edges_only=True)
    return graph, graph_ref


class TestPullReshapeThroughFQ(unittest.TestCase):

    def test_v7_weights_reshape(self):
        graph, graph_ref = graph_template([3, 8, 7, 7], [24, 1, 7, 7], [1, 1, 1, 1])

        PullReshapeThroughFQ().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_reshape_reducing_tensor_rank(self):
        graph, graph_ref = graph_template([3, 8, 7, 7], [24, 7, 7], [1, 1, 1])

        PullReshapeThroughFQ().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)


class TestV7ConvolutionWithGroupsResolver(unittest.TestCase):
    def test_v7_group_convolution_resolver(self):
        nodes = {
            **regular_op_with_shaped_data('input', [1, 3, 224, 224], {'type': 'Parameter'}),

            **valued_const_with_data('weights', np.ones([3, 8, 7, 7])),

            **valued_const_with_data('dim', int64_array([24, -1, 0, 0])),
            **regular_op_with_empty_data('reshape', {'type': 'Reshape'}),

            **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution', 'group': 3, 'output': 24}),

            **result(),
        }
        graph = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        V7ConvolutionWithGroupsResolver().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('weights', '0:reshape'),
            *connect('dim', '1:reshape'),
            *connect('reshape', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_v7_group_convolution_resolver_weight_are_in_the_right_layout(self):
        nodes = {
            **regular_op_with_shaped_data('input', [1, 3, 224, 224], {'type': 'Parameter'}),
            **valued_const_with_data('weights', np.ones([24, 1, 7, 7])),
            **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution', 'group': 3, 'output': 24}),
            **result(),
        }
        edges = [
            *connect('input', '0:convolution'),
            *connect('weights', '1:convolution'),
            *connect('convolution', 'output'),
        ]
        graph = build_graph(nodes, edges)
        V7ConvolutionWithGroupsResolver().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, edges)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_v7_group_convolution_resolver_depthwise_conv2d(self):
        nodes = {
            **regular_op_with_shaped_data('input', [1, 1, 224, 224], {'type': 'Parameter'}),

            **valued_const_with_data('weights', np.ones([1, 8, 7, 7])),

            **valued_const_with_data('dim', int64_array([8, -1, 0, 0])),
            **regular_op_with_empty_data('reshape', {'type': 'Reshape'}),

            **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution', 'group': 1, 'output': 8,
                                                                'op': 'DepthwiseConv2dNative'}),

            **result(),
        }
        graph = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        V7ConvolutionWithGroupsResolver().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('weights', '0:reshape'),
            *connect('dim', '1:reshape'),
            *connect('reshape', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)


class TestV10ConvolutionWithGroupsResolver(unittest.TestCase):

    @staticmethod
    def apply_transformation(graph):
        V10ConvolutionWithGroupsResolver().find_and_replace_pattern(graph)
        graph.clean_up()
        ShapeOfConstFolding().find_and_replace_pattern(graph)
        graph.clean_up()

    def test_v10_group_convolution_resolver(self):
        nodes = {
            **regular_op_with_shaped_data('input', [1, 3, 224, 224], {'type': 'Parameter'}),

            **valued_const_with_data('weights', np.ones([3, 8, 7, 7])),

            **valued_const_with_data('new_weights', np.ones([3, 8, 1, 7, 7])),

            **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution', 'group': 3, 'output': 24}),

            **result(),
        }
        graph = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        TestV10ConvolutionWithGroupsResolver.apply_transformation(graph)

        nodes['convolution']['type'] = 'GroupConvolution'
        del nodes['convolution']['group']

        graph_ref = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('new_weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_v10_group_convolution_resolver_depthwise_conv2d(self):
        nodes = {
            **regular_op_with_shaped_data('input', [1, 1, 224, 224], {'type': 'Parameter'}),

            **valued_const_with_data('weights', np.ones([1, 8, 7, 7])),

            **valued_const_with_data('new_weights', np.ones([1, 8, 1, 7, 7])),

            **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution', 'group': 1, 'output': 8,
                                                                'op': 'DepthwiseConv2dNative'}),

            **result(),
        }
        graph = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        TestV10ConvolutionWithGroupsResolver.apply_transformation(graph)

        nodes['convolution']['type'] = 'GroupConvolution'
        del nodes['convolution']['group']

        graph_ref = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('new_weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_v10_group_convolution_resolver_depthwise_conv2d_dynamic(self):
        nodes = {
            **regular_op_with_shaped_data('input', [-1, -1, -1, -1], {'type': 'Parameter'}),

            **valued_const_with_data('weights', np.ones([1, 8, 7, 7])),

            **valued_const_with_data('new_weights', np.ones([1, 8, 1, 7, 7])),

            **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution', 'group': 1, 'output': 8,
                                                                'op': 'DepthwiseConv2dNative'}),

            **result(),
        }
        graph = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        TestV10ConvolutionWithGroupsResolver.apply_transformation(graph)

        nodes['convolution']['type'] = 'GroupConvolution'
        del nodes['convolution']['group']

        graph_ref = build_graph(nodes, [
            *connect('input', '0:convolution'),
            *connect('new_weights', '1:convolution'),
            *connect('convolution', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)


