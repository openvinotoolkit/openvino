# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.extractors import FakeValue
from unit_tests.utils.graph import build_graph

nodes_attributes = {'conv_input': {'value': None, 'kind': 'data'},
                    'conv_node': {'type': 'Convolution', 'kind': 'op'},
                    'conv_weights': {'value': FakeValue(None), 'kind': 'data'},
                    'conv_output': {'value': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'}
                    }


class TestConvolutionPartialInfer(unittest.TestCase):
    def test_caffe_conv2d_infer(self):
        graph = build_graph(nodes_attributes,
                            [('conv_input', 'conv_node'),
                             ('conv_weights', 'conv_node'),
                             ('conv_node', 'conv_output'),
                             ('conv_output', 'op_output')
                             ],
                            {'conv_output': {'shape': None},
                             'conv_input': {'shape': np.array([1, 3, 227, 227])},
                             'conv_weights': {'shape': np.array([64, 3, 3, 3]),
                                              'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'conv_node': {'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
                                           'conv_pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                                           'dilation': np.array([1, 1, 1, 1]), 'bias_addable': True, 'bias_term': False,
                                           'output_spatial_shape': None, 'output_shape': None,
                                           'stride': np.array([1, 1, 1, 1]), 'group': 1,
                                           'kernel_spatial_idx': np.array([2, 3]),
                                           'input_feature_channel': 1,
                                           'output_feature_channel': 0,
                                           'output': 64, 'kernel_spatial': np.array([3, 3]),
                                           'spatial_dims': np.array([2, 3]), 'channel_dims': np.array([1]),
                                           'batch_dims': np.array([0])}
                             })

        conv_node = Node(graph, 'conv_node')
        Convolution.infer(conv_node)
        exp_shape = np.array([1, 64, 225, 225])
        res_shape = graph.node['conv_output']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_conv2d_dynamic_input_infer(self):
        graph = build_graph(nodes_attributes,
                            [('conv_input', 'conv_node'),
                             ('conv_weights', 'conv_node'),
                             ('conv_node', 'conv_output'),
                             ('conv_output', 'op_output')
                             ],
                            {'conv_output': {'shape': None},
                             'conv_input': {'shape': shape_array([1, 3, dynamic_dimension_value, 227])},
                             'conv_weights': {'shape': np.array([64, 3, 3, 3]),
                                              'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'conv_node': {'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
                                           'conv_pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                                           'dilation': np.array([1, 1, 1, 1]), 'bias_addable': True, 'bias_term': False,
                                           'output_spatial_shape': None, 'output_shape': None,
                                           'stride': np.array([1, 1, 1, 1]), 'group': 1,
                                           'kernel_spatial_idx': np.array([2, 3]),
                                           'input_feature_channel': 1,
                                           'output_feature_channel': 0,
                                           'output': 64, 'kernel_spatial': np.array([3, 3]),
                                           'spatial_dims': np.array([2, 3]), 'channel_dims': np.array([1]),
                                           'batch_dims': np.array([0])}
                             })

        conv_node = Node(graph, 'conv_node')
        Convolution.infer(conv_node)
        exp_shape = shape_array([1, 64, dynamic_dimension_value, 225])
        res_shape = graph.node['conv_output']['shape']
        self.assertTrue(strict_compare_tensors(exp_shape, res_shape))

    def test_caffe_conv2d_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('conv_input', 'conv_node'),
                             ('conv_weights', 'conv_node'),
                             ('conv_node', 'conv_output'),
                             ('conv_output', 'op_output')
                             ],
                            {'conv_output': {'shape': None},
                             'conv_input': {'shape': None},
                             'conv_weights': {'shape': None,
                                              'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'conv_node': {'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
                                           'conv_pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                                           'dilation': np.array([1, 1, 1, 1]), 'bias_addable': True, 'bias_term': False,
                                           'output_spatial_shape': None, 'output_shape': None,
                                           'stride': np.array([1, 1, 1, 1]), 'group': 1,
                                           'output': 64, 'kernel_spatial': np.array([3, 3]),
                                           'spatial_dims': np.array([2, 3]), 'channel_dims': np.array([1]),
                                           'batch_dims': np.array([0])}
                             })

        conv_node = Node(graph, 'conv_node')
        with self.assertRaisesRegex(Error, "Input data shape is None for node.*"):
            Convolution.infer(conv_node)

    def test_deconv_infer_ideal(self):
        graph = build_graph(nodes_attributes,
                            [('conv_input', 'conv_node'),
                             ('conv_weights', 'conv_node'),
                             ('conv_node', 'conv_output'),
                             ('conv_output', 'op_output')
                             ],
                            {'conv_output': {'shape': None},
                             'conv_input': {'shape': np.array([1, 21, 16, 16])},
                             'conv_weights': {'shape': np.array([1, 21, 4, 4]),
                                              'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'conv_node': {#'spatial_dims': np.array([2, 3]), 'batch_dims': np.array([0]),
                                           'channel_dims': np.array([1]), 'bias_addable': True, 'bias_term': False,
                                           'batch_dims': np.array([0]),
                                           'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
                                           'kernel_spatial': np.array([4, 4]), 'output_spatial_shape': None,
                                           'kernel_spatial_idx': np.array([2, 3]),
                                           'input_feature_channel': 1,
                                           'output_feature_channel': 0,
                                           'output_padding': np.array([0, 0, 1, 1]),
                                           'type': 'Deconvolution', 'output': 21, 'dilation': np.array([1, 1, 1, 1]),
                                           'group': 1, 'stride': np.array([1, 1, 2, 2]), 'output_shape': None}
                             })

        deconv_node = Node(graph, 'conv_node')

        Convolution.infer(deconv_node)
        res_shape = deconv_node['output_shape']
        exp_shape = np.array([1, 21, 35, 35])

        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        # Check that after double infer shape and pad attrs do not changes
        Convolution.infer(deconv_node)

        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_deconv_dynamic_infer_ideal(self):
        graph = build_graph(nodes_attributes,
                            [('conv_input', 'conv_node'),
                             ('conv_weights', 'conv_node'),
                             ('conv_node', 'conv_output'),
                             ('conv_output', 'op_output')
                             ],
                            {'conv_output': {'shape': None},
                             'conv_input': {'shape': shape_array([1, 21, dynamic_dimension_value, 16])},
                             'conv_weights': {'shape': np.array([1, 21, 4, 4]),
                                              'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'conv_node': {#'spatial_dims': np.array([2, 3]), 'batch_dims': np.array([0]),
                                           'channel_dims': np.array([1]), 'bias_addable': True, 'bias_term': False,
                                           'batch_dims': np.array([0]),
                                           'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
                                           'kernel_spatial': np.array([4, 4]), 'output_spatial_shape': None,
                                           'kernel_spatial_idx': np.array([2, 3]),
                                           'input_feature_channel': 1,
                                           'output_feature_channel': 0,
                                           'output_padding': np.array([0, 0, 1, 1]),
                                           'type': 'Deconvolution', 'output': 21, 'dilation': np.array([1, 1, 1, 1]),
                                           'group': 1, 'stride': np.array([1, 1, 2, 2]), 'output_shape': None}
                             })

        deconv_node = Node(graph, 'conv_node')

        Convolution.infer(deconv_node)
        res_shape = deconv_node['output_shape']
        exp_shape = shape_array([1, 21, dynamic_dimension_value, 35])

        self.assertTrue(strict_compare_tensors(exp_shape, res_shape))

        # Check that after double infer shape and pad attrs do not changes
        Convolution.infer(deconv_node)

        self.assertTrue(strict_compare_tensors(exp_shape, res_shape))

    def test_deconv_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('conv_input', 'conv_node'),
                             ('conv_weights', 'conv_node'),
                             ('conv_node', 'conv_output'),
                             ('conv_output', 'op_output')
                             ],
                            {'conv_output': {'shape': None},
                             'conv_input': {'shape': None},
                             'conv_weights': {'shape': np.array([1, 21, 16, 16]),
                                              'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'conv_node': {'spatial_dims': np.array([2, 3]), 'batch_dims': np.array([0]),
                                           'channel_dims': np.array([1]),
                                           'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
                                           'kernel_spatial': np.array([4, 4]), 'output_spatial_shape': None,
                                           'kernel_spatial_idx': np.array([2, 3]),
                                           'input_feature_channel': 1,
                                           'output_feature_channel': 0,
                                           'type': 'Deconvolution', 'output': 21, 'dilation': np.array([1, 1, 1, 1]),
                                           'group': 1, 'stride': np.array([1, 1, 2, 2]), 'output_shape': None}
                             })

        deconv_node = Node(graph, 'conv_node')
        with self.assertRaisesRegex(Error, "Input data shape is None for node.*"):
            Convolution.infer(deconv_node)

    def test_conv_infer_set_default_attrs_nchw(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('conv_input', 'conv_node'),
                                ('conv_weights', 'conv_node'),
                                ('conv_node', 'conv_output'),
                                ('conv_output', 'op_output')
                            ],
                            {
                                'conv_output': {
                                    'shape': None
                                },
                                'conv_input': {
                                    'shape': int64_array([1, 3, 224, 224])
                                },
                                'conv_weights': {
                                    'shape': int64_array([3, 64, 7, 7]),
                                    'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']
                                },
                                'conv_node': {
                                    'type': 'Convolution',
                                    'bias_term': None,
                                    'stride': None,
                                    'dilation': None,

                                    'batch_dims': int64_array([0]),
                                    'channel_dims': int64_array([1]),

                                    'output_spatial_shape': None,

                                    'input_feature_channel': 0,
                                    'output_feature_channel': 1,

                                    'group': 1,
                                    'output_shape': None,
                                    'layout': 'NCHW'
                                }
                            })

        conv_node = Node(graph, 'conv_node')
        conv_output = Node(graph, 'conv_output')

        Convolution.infer(conv_node)

        # Check bias_term attribute
        self.assertTrue(conv_node.has_valid('bias_term'))
        self.assertTrue(not conv_node.bias_term)
        # Check kernel_spatial_idx attr detection
        self.assertTrue(conv_node.has_valid('kernel_spatial_idx'))
        self.assertTrue(np.array_equal(int64_array([2, 3]), conv_node.kernel_spatial_idx))
        # Check spatial_dims attr detection
        self.assertTrue(conv_node.has_valid('spatial_dims'))
        self.assertTrue(np.array_equal(int64_array([2, 3]), conv_node.spatial_dims))
        # Check kernel_spatial attr detection
        self.assertTrue(conv_node.has_valid('kernel_spatial'))
        self.assertTrue(np.array_equal(int64_array([7, 7]), conv_node.kernel_spatial))
        # Check output attribute
        self.assertTrue(conv_node.has_valid('output'))
        self.assertEqual(64, conv_node.output)
        # Check dilation value. Should be set to default
        self.assertTrue(conv_node.has_valid('dilation'))
        self.assertTrue(np.array_equal(int64_array([1, 1, 1, 1]), conv_node.dilation))
        # Check stride value. Should be set to default
        self.assertTrue(conv_node.has_valid('stride'))
        self.assertTrue(np.array_equal(int64_array([1, 1, 1, 1]), conv_node.stride))
        # Check pad value. Should be set to default
        self.assertTrue(conv_node.has_valid('pad'))
        self.assertTrue(np.array_equal(int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]), conv_node.pad))
        # Check pad_spatial_shape
        self.assertTrue(conv_node.has_valid('pad_spatial_shape'))
        self.assertTrue(np.array_equal(int64_array([[0, 0], [0, 0]]), conv_node.pad_spatial_shape))
        # Check resulting output shape
        self.assertTrue(np.array_equal(int64_array([1, 64, 218, 218]), conv_output.shape))

    def test_conv_infer_set_default_attrs_nhwc(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('conv_input', 'conv_node'),
                                ('conv_weights', 'conv_node'),
                                ('conv_node', 'conv_output'),
                                ('conv_output', 'op_output')
                            ],
                            {
                                'conv_output': {
                                    'shape': None
                                },
                                'conv_input': {
                                    'shape': int64_array([1, 224, 224, 3])
                                },
                                'conv_weights': {
                                    'shape': int64_array([3, 64, 7, 7]),
                                    'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']
                                },
                                'conv_node': {
                                    'type': 'Convolution',
                                    'bias_term': None,
                                    'stride': None,
                                    'dilation': None,

                                    'batch_dims': int64_array([0]),
                                    'channel_dims': int64_array([3]),

                                    'output_spatial_shape': None,

                                    'input_feature_channel': 0,
                                    'output_feature_channel': 1,

                                    'group': 1,
                                    'output_shape': None,
                                    'layout': 'NHWC'
                                }
                            })

        conv_node = Node(graph, 'conv_node')
        conv_output = Node(graph, 'conv_output')

        Convolution.infer(conv_node)

        # Check bias_term attribute
        self.assertTrue(conv_node.has_valid('bias_term'))
        self.assertTrue(not conv_node.bias_term)
        # Check kernel_spatial_idx attr detection
        self.assertTrue(conv_node.has_valid('kernel_spatial_idx'))
        self.assertTrue(np.array_equal(int64_array([2, 3]), conv_node.kernel_spatial_idx))
        # Check spatial_dims attr detection
        self.assertTrue(conv_node.has_valid('spatial_dims'))
        self.assertTrue(np.array_equal(int64_array([1, 2]), conv_node.spatial_dims))
        # Check kernel_spatial attr detection
        self.assertTrue(conv_node.has_valid('kernel_spatial'))
        self.assertTrue(np.array_equal(int64_array([7, 7]), conv_node.kernel_spatial))
        # Check output attribute
        self.assertTrue(conv_node.has_valid('output'))
        self.assertEqual(64, conv_node.output)
        # Check dilation value. Should be set to default
        self.assertTrue(conv_node.has_valid('dilation'))
        self.assertTrue(np.array_equal(int64_array([1, 1, 1, 1]), conv_node.dilation))
        # Check stride value. Should be set to default
        self.assertTrue(conv_node.has_valid('stride'))
        self.assertTrue(np.array_equal(int64_array([1, 1, 1, 1]), conv_node.stride))
        # Check pad value. Should be set to default
        self.assertTrue(conv_node.has_valid('pad'))
        self.assertTrue(np.array_equal(int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]), conv_node.pad))
        # Check pad_spatial_shape
        self.assertTrue(conv_node.has_valid('pad_spatial_shape'))
        self.assertTrue(np.array_equal(int64_array([[0, 0], [0, 0]]), conv_node.pad_spatial_shape))
        # Check resulting output shape
        self.assertTrue(np.array_equal(int64_array([1, 218, 218, 64]), conv_output.shape))

    def test_conv_infer_3D_convolution(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('conv_input', 'conv_node'),
                                ('conv_weights', 'conv_node'),
                                ('conv_node', 'conv_output'),
                                ('conv_output', 'op_output')
                            ],
                            {
                                'conv_output': {
                                    'shape': None
                                },
                                'conv_input': {
                                    'shape': int64_array([1, 3, 16, 224, 224])
                                },
                                'conv_weights': {
                                    'shape': int64_array([3, 64, 1, 7, 7]),
                                    'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']
                                },
                                'conv_node': {
                                    'type': 'Convolution',
                                    'bias_term': None,
                                    'stride': None,
                                    'dilation': None,

                                    'batch_dims': int64_array([0]),
                                    'channel_dims': int64_array([1]),

                                    'output_spatial_shape': None,

                                    'input_feature_channel': 0,
                                    'output_feature_channel': 1,

                                    'group': 1,
                                    'output_shape': None,
                                    'layout': 'NCHW'
                                }
                            })

        conv_node = Node(graph, 'conv_node')
        conv_output = Node(graph, 'conv_output')

        Convolution.infer(conv_node)

        # Check bias_term attribute
        self.assertTrue(conv_node.has_valid('bias_term'))
        self.assertTrue(not conv_node.bias_term)
        # Check kernel_spatial_idx attr detection
        self.assertTrue(conv_node.has_valid('kernel_spatial_idx'))
        self.assertTrue(np.array_equal(int64_array([2, 3, 4]), conv_node.kernel_spatial_idx))
        # Check spatial_dims attr detection
        self.assertTrue(conv_node.has_valid('spatial_dims'))
        self.assertTrue(np.array_equal(int64_array([2, 3, 4]), conv_node.spatial_dims))
        # Check kernel_spatial attr detection
        self.assertTrue(conv_node.has_valid('kernel_spatial'))
        self.assertTrue(np.array_equal(int64_array([1, 7, 7]), conv_node.kernel_spatial))
        # Check output attribute
        self.assertTrue(conv_node.has_valid('output'))
        self.assertEqual(64, conv_node.output)
        # Check dilation value. Should be set to default
        self.assertTrue(conv_node.has_valid('dilation'))
        self.assertTrue(np.array_equal(int64_array([1, 1, 1, 1, 1]), conv_node.dilation))
        # Check stride value. Should be set to default
        self.assertTrue(conv_node.has_valid('stride'))
        self.assertTrue(np.array_equal(int64_array([1, 1, 1, 1, 1]), conv_node.stride))
        # Check pad value. Should be set to default
        self.assertTrue(conv_node.has_valid('pad'))
        self.assertTrue(np.array_equal(int64_array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]), conv_node.pad))
        # Check pad_spatial_shape
        self.assertTrue(conv_node.has_valid('pad_spatial_shape'))
        self.assertTrue(np.array_equal(int64_array([[0, 0], [0, 0], [0, 0]]), conv_node.pad_spatial_shape))
        # Check resulting output shape
        self.assertTrue(np.array_equal(int64_array([1, 64, 16, 218, 218]), conv_output.shape))

    def test_caffe_conv2d_infer_wrong_input_shape(self):
        graph = build_graph(nodes_attributes,
                            [('conv_input', 'conv_node'),
                             ('conv_weights', 'conv_node'),
                             ('conv_node', 'conv_output'),
                             ('conv_output', 'op_output')
                             ],
                            {'conv_output': {'shape': None},
                             'conv_input': {'shape': np.array([1, 3, 1, 1])},
                             'conv_weights': {'shape': np.array([64, 3, 3, 3]),
                                              'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
                             'conv_node': {'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
                                           'conv_pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
                                           'dilation': np.array([1, 1, 1, 1]), 'bias_addable': True, 'bias_term': False,
                                           'output_spatial_shape': None, 'output_shape': None,
                                           'stride': np.array([1, 1, 1, 1]), 'group': 1,
                                           'kernel_spatial_idx': np.array([2, 3]),
                                           'input_feature_channel': 1,
                                           'output_feature_channel': 0,
                                           'output': 64, 'kernel_spatial': np.array([3, 3]),
                                           'spatial_dims': np.array([2, 3]), 'channel_dims': np.array([1]),
                                           'batch_dims': np.array([0])}
                             })

        conv_node = Node(graph, 'conv_node')
        with self.assertRaises(Error):
            Convolution.infer(conv_node)
