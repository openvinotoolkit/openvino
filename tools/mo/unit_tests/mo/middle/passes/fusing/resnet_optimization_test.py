# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.eltwise import eltwise_infer
from openvino.tools.mo.middle.passes.fusing.resnet_optimization import stride_optimization
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.ops.pooling import Pooling
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

max_elt_lambda = lambda node: eltwise_infer(node, lambda a, b: np.maximum(a, b))

nodes_attributes = {
    # Placeholders
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Concat1 operation
    'eltwise_1': {'type': 'Maximum', 'kind': 'op', 'op': 'Maximum', 'infer': max_elt_lambda},
    'eltwise_1_data': {'name': 'eltwise_1_data', 'value': None, 'shape': None, 'kind': 'data'},
    # Convolutions
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NCHW',
               'output_spatial_shape': None, 'output_shape': None, 'bias_term': True, 'group': 1,
               'spatial_dims': np.array([2, 3]),
               'channel_dims': np.array([1]), 'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
               'dilation': np.array([1, 1, 1, 1]),
               'batch_dims': np.array([0]), 'infer': Convolution.infer,
               'kernel_spatial_idx': np.array([2, 3], dtype=np.int64), 'input_feature_channel': 1,
               'output_feature_channel': 0, },
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data',
                 'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NCHW',
               'output_spatial_shape': None, 'output_shape': None, 'bias_term': True, 'group': 1,
               'spatial_dims': np.array([2, 3]),
               'channel_dims': np.array([1]), 'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
               'dilation': np.array([1, 1, 1, 1]),
               'batch_dims': np.array([0]), 'infer': Convolution.infer,
               'kernel_spatial_idx': np.array([2, 3], dtype=np.int64), 'input_feature_channel': 1,
               'output_feature_channel': 0, },
    'conv_2_w': {'value': None, 'shape': None, 'kind': 'data',
                 'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
    'conv_2_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    'conv_3': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NCHW',
               'output_spatial_shape': None, 'output_shape': None, 'bias_term': True, 'group': 1,
               'spatial_dims': np.array([2, 3]),
               'channel_dims': np.array([1]), 'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
               'dilation': np.array([1, 1, 1, 1]),
               'batch_dims': np.array([0]), 'infer': Convolution.infer,
               'kernel_spatial_idx': np.array([2, 3], dtype=np.int64), 'input_feature_channel': 1,
               'output_feature_channel': 0, },
    'conv_3_w': {'value': None, 'shape': None, 'kind': 'data',
                 'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
    'conv_3_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_3_data': {'value': None, 'shape': None, 'kind': 'data'},

    'conv_4': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NCHW',
               'output_spatial_shape': None, 'output_shape': None, 'bias_term': True, 'group': 1,
               'spatial_dims': np.array([2, 3]),
               'channel_dims': np.array([1]), 'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
               'dilation': np.array([1, 1, 1, 1]),
               'batch_dims': np.array([0]), 'infer': Convolution.infer,
               'kernel_spatial_idx': np.array([2, 3], dtype=np.int64), 'input_feature_channel': 1,
               'output_feature_channel': 0, },
    'conv_4_w': {'value': None, 'shape': None, 'kind': 'data',
                 'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
    'conv_4_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_4_data': {'value': None, 'shape': None, 'kind': 'data'},

    'conv_5': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NCHW',
               'output_spatial_shape': None, 'output_shape': None, 'bias_term': True, 'group': 1,
               'spatial_dims': np.array([2, 3]),
               'channel_dims': np.array([1]), 'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
               'dilation': np.array([1, 1, 1, 1]),
               'batch_dims': np.array([0]), 'infer': Convolution.infer,
               'kernel_spatial_idx': np.array([2, 3], dtype=np.int64), 'input_feature_channel': 1,
               'output_feature_channel': 0, },
    'conv_5_w': {'value': None, 'shape': None, 'kind': 'data',
                 'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis']},
    'conv_5_b': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_5_data': {'value': None, 'shape': None, 'kind': 'data'},
    # ReLU
    'relu_1': {'shape': None, 'type': 'ReLU', 'kind': 'op', 'op': 'ReLU', 'infer': copy_shape_infer},
    'relu_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'relu_2': {'shape': None, 'type': 'ReLU', 'kind': 'op', 'op': 'ReLU', 'infer': copy_shape_infer},
    'relu_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'relu_3': {'shape': None, 'type': 'ReLU', 'kind': 'op', 'op': 'ReLU', 'infer': copy_shape_infer},
    'relu_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Pooling
    'pool_1': {'type': 'Pooling', 'kind': 'op', 'op': 'Pooling',
               'spatial_dims': np.array([2, 3]),
               'pad_spatial_shape': np.array([[0, 0], [0, 0]]),
               'infer': Pooling.infer},
    'pool_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


# In description of unit tests below will be used next syntax: Operation(NxM,XxY), where NxM - kernel size, XxY - stride
class ResnetOptimizationTests(unittest.TestCase):
    # Pl->Conv(1x1,1x1)->Conv(1x1,2x2) => Pl->Conv(1x1,2x2)->Conv(1x1,1x1)
    def test_resnet_optimization_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'conv_2'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_1': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 1, 1]),
                                        'output': np.array([3]), },
                             'conv_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_2_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_2': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_2_data': {'shape': np.array([1, 3, 112, 112])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'conv_2'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                                 'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_1': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 2, 2]),
                                            'output': np.array([3]), },
                                 'conv_1_data': {'shape': np.array([1, 3, 112, 112])},

                                 'conv_2_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_2': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 1, 1]),
                                            'output': np.array([3]), },
                                 'conv_2_data': {'shape': np.array([1, 3, 112, 112])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph_ref.graph['layout'] = 'NCHW'

        stride_optimization(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Pl->Conv(3x3,2x2)->Conv(1x1,2x2) => Pl->Conv(3x3,4x4)->Conv(1x1,1x1)
    def test_resnet_optimization_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'conv_2'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_1': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_1_data': {'shape': np.array([1, 3, 112, 112])},

                             'conv_2_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_2': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_2_data': {'shape': np.array([1, 3, 56, 56])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'conv_2'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                                 'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_1': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 4, 4]),
                                            'output': np.array([3]), },
                                 'conv_1_data': {'shape': np.array([1, 3, 56, 56])},

                                 'conv_2_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_2': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 1, 1]),
                                            'output': np.array([3]), },
                                 'conv_2_data': {'shape': np.array([1, 3, 56, 56])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph_ref.graph['layout'] = 'NCHW'

        stride_optimization(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Pl->Conv(3x3,2x2)->Conv(3x3,2x2) => Same
    def test_resnet_optimization_3(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'conv_2'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_1_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                             'conv_1': {'kernel_spatial': np.array([3, 3]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_1_data': {'shape': np.array([1, 3, 112, 112])},

                             'conv_2_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                             'conv_2': {'kernel_spatial': np.array([3, 3]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_2_data': {'shape': np.array([1, 3, 56, 56])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'conv_2'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                                 'conv_1_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                                 'conv_1': {'kernel_spatial': np.array([3, 3]),
                                            'stride': np.array([1, 1, 2, 2]),
                                            'output': np.array([3]), },
                                 'conv_1_data': {'shape': np.array([1, 3, 112, 112])},

                                 'conv_2_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                                 'conv_2': {'kernel_spatial': np.array([3, 3]),
                                            'stride': np.array([1, 1, 2, 2]),
                                            'output': np.array([3]), },
                                 'conv_2_data': {'shape': np.array([1, 3, 56, 56])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph_ref.graph['layout'] = 'NCHW'

        stride_optimization(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Pl--->Conv(3x3,2x2)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(3x3,4x4)->ReLU--->Eltwise-->Conv(1x1,1x1)
    #   `-->Conv(3x3,2x2)->ReLU---`                             `-->Conv(3x3,4x4)->ReLU---`
    def test_resnet_optimization_4(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'relu_1'),
                             ('relu_1', 'relu_1_data'),

                             ('placeholder_1_data', 'conv_2'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),
                             ('conv_2_data', 'relu_2'),
                             ('relu_2', 'relu_2_data'),

                             ('relu_1_data', 'eltwise_1'),
                             ('relu_2_data', 'eltwise_1'),

                             ('eltwise_1', 'eltwise_1_data'),
                             ('eltwise_1_data', 'conv_3'),
                             ('conv_3_w', 'conv_3'),
                             ('conv_3_b', 'conv_3'),
                             ('conv_3', 'conv_3_data'),

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_1_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                             'conv_1': {'kernel_spatial': np.array([3, 3]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_1_data': {'shape': np.array([1, 3, 112, 112])},
                             'relu_1_data': {'shape': np.array([1, 3, 112, 112])},

                             'conv_2_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                             'conv_2': {'kernel_spatial': np.array([3, 3]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_2_data': {'shape': np.array([1, 3, 112, 112])},
                             'relu_2_data': {'shape': np.array([1, 3, 112, 112])},

                             'eltwise_1_data': {'shape': np.array([1, 3, 112, 112])},

                             'conv_3_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_3': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_3_data': {'shape': np.array([1, 3, 56, 56])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'relu_1'),
                                 ('relu_1', 'relu_1_data'),

                                 ('placeholder_1_data', 'conv_2'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),
                                 ('conv_2_data', 'relu_2'),
                                 ('relu_2', 'relu_2_data'),

                                 ('relu_1_data', 'eltwise_1'),
                                 ('relu_2_data', 'eltwise_1'),

                                 ('eltwise_1', 'eltwise_1_data'),
                                 ('eltwise_1_data', 'conv_3'),
                                 ('conv_3_w', 'conv_3'),
                                 ('conv_3_b', 'conv_3'),
                                 ('conv_3', 'conv_3_data'),

                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                                 'conv_1_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                                 'conv_1': {'kernel_spatial': np.array([3, 3]),
                                            'stride': np.array([1, 1, 4, 4]),
                                            'output': np.array([3])},
                                 'conv_1_data': {'shape': np.array([1, 3, 56, 56])},
                                 'relu_1_data': {'shape': np.array([1, 3, 56, 56])},

                                 'conv_2_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                                 'conv_2': {'kernel_spatial': np.array([3, 3]),
                                            'stride': np.array([1, 1, 4, 4]),
                                            'output': np.array([3])},
                                 'conv_2_data': {'shape': np.array([1, 3, 56, 56])},
                                 'relu_2_data': {'shape': np.array([1, 3, 56, 56])},

                                 'eltwise_1_data': {'shape': np.array([1, 3, 56, 56])},

                                 'conv_3_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_3': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 1, 1]),
                                            'output': np.array([3])},
                                 'conv_3_data': {'shape': np.array([1, 3, 56, 56])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph_ref.graph['layout'] = 'NCHW'

        stride_optimization(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_3_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Pl--->Conv(1x1,1x1)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(1x1,2x2)->ReLU--->Eltwise-->Conv(1x1,1x1)
    #   `----------------->ReLU---`                             `-->Pool(1x1,2x2)->ReLU---`
    def test_resnet_optimization_5(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),
                             ('conv_1_data', 'relu_1'),
                             ('relu_1', 'relu_1_data'),

                             ('placeholder_1_data', 'relu_2'),
                             ('relu_2', 'relu_2_data'),

                             ('relu_1_data', 'eltwise_1'),
                             ('relu_2_data', 'eltwise_1'),

                             ('eltwise_1', 'eltwise_1_data'),
                             ('eltwise_1_data', 'conv_3'),
                             ('conv_3_w', 'conv_3'),
                             ('conv_3_b', 'conv_3'),
                             ('conv_3', 'conv_3_data'),

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_1': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 1, 1]),
                                        'output': np.array([3]), },
                             'conv_1_data': {'shape': np.array([1, 3, 224, 224])},
                             'relu_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'relu_2_data': {'shape': np.array([1, 3, 224, 224])},

                             'eltwise_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_3_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_3': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_3_data': {'shape': np.array([1, 3, 112, 112])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),
                                 ('conv_1_data', 'relu_1'),
                                 ('relu_1', 'relu_1_data'),

                                 ('placeholder_1_data', 'pool_1'),
                                 ('pool_1', 'pool_1_data'),
                                 ('pool_1_data', 'relu_2'),
                                 ('relu_2', 'relu_2_data'),

                                 ('relu_1_data', 'eltwise_1'),
                                 ('relu_2_data', 'eltwise_1'),

                                 ('eltwise_1', 'eltwise_1_data'),
                                 ('eltwise_1_data', 'conv_3'),
                                 ('conv_3_w', 'conv_3'),
                                 ('conv_3_b', 'conv_3'),
                                 ('conv_3', 'conv_3_data'),

                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                                 'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_1': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 2, 2]),
                                            'output': np.array([3])},
                                 'conv_1_data': {'shape': np.array([1, 3, 112, 112])},
                                 'relu_1_data': {'shape': np.array([1, 3, 112, 112])},

                                 'pool_1': {'stride': np.array([1, 1, 2, 2])},
                                 'pool_1_data': {'shape': np.array([1, 3, 112, 112])},
                                 'relu_2_data': {'shape': np.array([1, 3, 112, 112])},

                                 'eltwise_1_data': {'shape': np.array([1, 3, 112, 112])},

                                 'conv_3_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_3': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 1, 1]),
                                            'output': np.array([3])},
                                 'conv_3_data': {'shape': np.array([1, 3, 112, 112])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph_ref.graph['layout'] = 'NCHW'

        stride_optimization(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_3_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # Pl->Conv(1x1,1x1)->Conv(1x1,2x2)->Conv(3x3,1x1)->Conv(1x1,2x2)
    #       =>
    # Pl->Conv(1x1,2x2)->Conv(1x1,1x1)->Conv(3x3,2x2)->Conv(1x1,1x1)
    def test_resnet_optimization_6(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'conv_1'),
                             ('conv_1_w', 'conv_1'),
                             ('conv_1_b', 'conv_1'),
                             ('conv_1', 'conv_1_data'),

                             ('conv_1_data', 'conv_2'),
                             ('conv_2_w', 'conv_2'),
                             ('conv_2_b', 'conv_2'),
                             ('conv_2', 'conv_2_data'),

                             ('conv_2_data', 'conv_3'),
                             ('conv_3_w', 'conv_3'),
                             ('conv_3_b', 'conv_3'),
                             ('conv_3', 'conv_3_data'),

                             ('conv_3_data', 'conv_4'),
                             ('conv_4_w', 'conv_4'),
                             ('conv_4_b', 'conv_4'),
                             ('conv_4', 'conv_4_data'),

                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_1': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 1, 1]),
                                        'output': np.array([3]), },
                             'conv_1_data': {'shape': np.array([1, 3, 224, 224])},

                             'conv_2_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_2': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_2_data': {'shape': np.array([1, 3, 112, 112])},

                             'conv_3_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                             'conv_3': {'kernel_spatial': np.array([3, 3]),
                                        'stride': np.array([1, 1, 1, 1]),
                                        'output': np.array([3]), },
                             'conv_3_data': {'shape': np.array([1, 3, 110, 110])},

                             'conv_4_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                             'conv_4': {'kernel_spatial': np.array([1, 1]),
                                        'stride': np.array([1, 1, 2, 2]),
                                        'output': np.array([3]), },
                             'conv_4_data': {'shape': np.array([1, 3, 55, 55])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'conv_1'),
                                 ('conv_1_w', 'conv_1'),
                                 ('conv_1_b', 'conv_1'),
                                 ('conv_1', 'conv_1_data'),

                                 ('conv_1_data', 'conv_2'),
                                 ('conv_2_w', 'conv_2'),
                                 ('conv_2_b', 'conv_2'),
                                 ('conv_2', 'conv_2_data'),

                                 ('conv_2_data', 'conv_3'),
                                 ('conv_3_w', 'conv_3'),
                                 ('conv_3_b', 'conv_3'),
                                 ('conv_3', 'conv_3_data'),

                                 ('conv_3_data', 'conv_4'),
                                 ('conv_4_w', 'conv_4'),
                                 ('conv_4_b', 'conv_4'),
                                 ('conv_4', 'conv_4_data'),

                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},

                                 'conv_1_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_1': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 2, 2]),
                                            'output': np.array([3])},
                                 'conv_1_data': {'shape': np.array([1, 3, 112, 112])},

                                 'conv_2_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_2': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 1, 1]),
                                            'output': np.array([3])},
                                 'conv_2_data': {'shape': np.array([1, 3, 112, 112])},

                                 'conv_3_w': {'value': np.zeros([3, 3, 3, 3]), 'shape': np.array([3, 3, 3, 3])},
                                 'conv_3': {'kernel_spatial': np.array([3, 3]),
                                            'stride': np.array([1, 1, 2, 2]),
                                            'output': np.array([3])},
                                 'conv_3_data': {'shape': np.array([1, 3, 55, 55])},

                                 'conv_4_w': {'value': np.zeros([3, 3, 1, 1]), 'shape': np.array([3, 3, 1, 1])},
                                 'conv_4': {'kernel_spatial': np.array([1, 1]),
                                            'stride': np.array([1, 1, 1, 1]),
                                            'output': np.array([3])},
                                 'conv_4_data': {'shape': np.array([1, 3, 55, 55])},
                                 },
                                nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph_ref.graph['layout'] = 'NCHW'

        stride_optimization(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_4_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
