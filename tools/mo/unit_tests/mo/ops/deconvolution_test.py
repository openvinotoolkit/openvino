# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.deconv_ext import get_conv_backprop_groups
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.deconvolution import Deconvolution
from unit_tests.utils.graph import build_graph

nodes_attributes = {'deconv_input': {'value': None, 'kind': 'data'},
                    'deconv_weights': {'value': None, 'kind': 'data'},
                    'deconv_output_shape': {'value': None, 'kind': 'data'},
                    'deconv_node': {'type': 'Deconvolution', 'op': 'Deconvolution', 'kind': 'op'},
                    'deconv_output': {'value': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'}
                    }


def create_deconv_graph(input_shape: int64_array, weights_shape: int64_array, output_shape: int64_array):
    graph = build_graph(nodes_attributes,
                        [('deconv_input', 'deconv_node'),
                         ('deconv_weights', 'deconv_node'),
                         ('deconv_output_shape', 'deconv_node'),
                         ('deconv_node', 'deconv_output'),
                         ('deconv_output', 'op_output')
                         ],
                        {'deconv_input': {'shape': input_shape},
                         'deconv_weights': {'shape': weights_shape,
                                            'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                         'deconv_output_shape': {'value': output_shape},
                         'deconv_node': {'channel_dims': int64_array([1]),
                                         'batch_dims': int64_array([0]),
                                         'spatial_dims': int64_array([2, 3]),
                                         'pad_spatial_shape': int64_array([[0, 0], [0, 0]]),
                                         'kernel_spatial': int64_array([4, 4]),
                                         'kernel_spatial_idx': int64_array([2, 3]),
                                         'input_feature_channel': 0,
                                         'output_feature_channel': 1,
                                         'auto_pad': 'same_lower',
                                         'output_padding': int64_array([0, 0, 1, 1]),
                                         'type': 'Deconvolution',
                                         'dilation': int64_array([1, 1, 1, 1]),
                                         'stride': int64_array([1, 1, 2, 2]),
                                         'pad': None,
                                         'output': None,
                                         'output_shape': None,
                                         'get_group': get_conv_backprop_groups},
                         'deconv_output': {'shape': None},
                         })
    return graph


class TestConvolutionPartialInfer(unittest.TestCase):
    def test_deconv_infer_one_group(self):
        graph = create_deconv_graph(int64_array([1, 21, 18, 18]), int64_array([21, 50, 4, 4]),
                                    int64_array([1, 50, 35, 35]))

        Deconvolution.infer(Node(graph, 'deconv_node'))
        res_shape = graph.node['deconv_output']['shape']
        exp_shape = np.array([1, 50, 35, 35])

        res_group = graph.node['deconv_node']['group']
        exp_group = int64_array([1])

        self.assertTrue(np.array_equal(exp_shape, res_shape),
                        'values do not match expected: {} and computed: {}'.format(exp_shape, res_shape))

        self.assertTrue(np.array_equal(exp_group, res_group),
                        'group number values do not match expected: {} and computed: {}'.format(exp_group, res_group))

    def test_deconv_infer_several_groups(self):
        graph = create_deconv_graph(int64_array([1, 21, 18, 18]), int64_array([21, 50, 4, 4]),
                                    int64_array([1, 350, 35, 35]))

        Deconvolution.infer(Node(graph, 'deconv_node'))
        res_shape = graph.node['deconv_output']['shape']
        exp_shape = np.array([1, 350, 35, 35])

        res_group = graph.node['deconv_node']['group']
        exp_group = int64_array([7])

        self.assertTrue(np.array_equal(exp_shape, res_shape),
                        'values do not match expected: {} and computed: {}'.format(exp_shape, res_shape))

        self.assertTrue(np.array_equal(exp_group, res_group),
                        'group number values do not match expected: {} and computed: {}'.format(exp_group, res_group))
