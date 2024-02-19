# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.pooling import Pooling
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'data'},
                    'pool': {'type': 'Pooling', 'value': None, 'kind': 'op'},
                    'node_2': {'value': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


class TestPoolingPartialInfer(unittest.TestCase):
    def test_pooling_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'pool'),
                             ('pool', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'pool': {'window': np.array([1, 1, 1, 1]), 'stride': np.array([1, 1, 2, 2]),
                                      'pad': np.array([[0, 0], [0, 0], [3, 3], [3, 3]]),
                                      'pad_spatial_shape': np.array([[3, 3], [3, 3]]),
                                      'pool_method': 'avg', 'exclude_pad': False, 'global_pool': False,
                                      'output_spatial_shape': None, 'output_shape': None,
                                      'kernel_spatial': np.array([3, 3]), 'spatial_dims': np.array([2, 3]),
                                      'channel_dims': np.array([1]), 'batch_dims': np.array([0]),
                                      'pooling_convention': 'full'}
                             })

        pool_node = Node(graph, 'pool')

        Pooling.infer(pool_node)
        exp_shape = np.array([1, 3, 131, 131])
        res_shape = graph.node['node_2']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_pooling_dynamic_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'pool'),
                             ('pool', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': shape_array([1, dynamic_dimension_value, dynamic_dimension_value,
                                                              256])},
                             'pool': {'window': np.array([1, 1, 1, 1]), 'stride': np.array([1, 1, 2, 2]),
                                      'pad': np.array([[0, 0], [0, 0], [3, 3], [3, 3]]),
                                      'pad_spatial_shape': np.array([[3, 3], [3, 3]]),
                                      'pool_method': 'avg', 'exclude_pad': False, 'global_pool': False,
                                      'output_spatial_shape': None, 'output_shape': None,
                                      'kernel_spatial': np.array([3, 3]), 'spatial_dims': np.array([2, 3]),
                                      'channel_dims': np.array([1]), 'batch_dims': np.array([0]),
                                      'pooling_convention': 'full'}
                             })

        pool_node = Node(graph, 'pool')

        Pooling.infer(pool_node)
        exp_shape = shape_array([1, dynamic_dimension_value, dynamic_dimension_value, 131])
        res_shape = graph.node['node_2']['shape']
        self.assertTrue(strict_compare_tensors(exp_shape, res_shape))

    def test_pooling_infer_decrement_input_spatial(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'pool'),
                             ('pool', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 224, 224])},
                             'pool': {'window': np.array([1, 1, 1, 1]), 'stride': np.array([1, 1, 3, 3]),
                                      'pad': np.array([[0, 0], [0, 0], [3, 3], [3, 3]]),
                                      'pad_spatial_shape': np.array([[1, 1], [1, 1]]),
                                      'pool_method': 'avg', 'exclude_pad': False, 'global_pool': False,
                                      'output_spatial_shape': None, 'output_shape': None,
                                      'kernel_spatial': np.array([3, 3]), 'spatial_dims': np.array([2, 3]),
                                      'channel_dims': np.array([1]), 'batch_dims': np.array([0]),
                                      'pooling_convention': 'full'}
                             })

        pool_node = Node(graph, 'pool')

        Pooling.infer(pool_node)
        exp_shape = np.array([1, 3, 75, 75])
        res_shape = graph.node['node_2']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_pooling_infer_no_convention(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'pool'),
                             ('pool', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'pool': {'window': np.array([1, 1, 1, 1]), 'stride': np.array([1, 1, 2, 2]),
                                      'pad': np.array([[0, 0], [0, 0], [3, 3], [3, 3]]),
                                      'pad_spatial_shape': np.array([[3, 3], [3, 3]]),
                                      'pool_method': 'avg', 'exclude_pad': False, 'global_pool': False,
                                      'output_spatial_shape': None, 'output_shape': None,
                                      'kernel_spatial': np.array([3, 3]), 'spatial_dims': np.array([2, 3]),
                                      'channel_dims': np.array([1]), 'batch_dims': np.array([0])}
                             })

        pool_node = Node(graph, 'pool')

        Pooling.infer(pool_node)
        exp_shape = np.array([1, 3, 130, 130])
        res_shape = graph.node['node_2']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_pooling_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'pool'),
                             ('pool', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': None},
                             'pool': {'window': np.array([1, 1, 1, 1]), 'stride': np.array([1, 1, 2, 2]),
                                      'pad': np.array([[0, 0], [0, 0], [3, 3], [3, 3]]),
                                      'pad_spatial_shape': np.array([[3, 3], [3, 3]]),
                                      'pool_method': 'avg', 'exclude_pad': False,
                                      'output_spatial_shape': None, 'output_shape': None,
                                      'kernel_spatial': np.array([3, 3]), 'spatial_dims': np.array([2, 3]),
                                      'channel_dims': np.array([1]), 'batch_dims': np.array([0]),
                                      'pooling_convention': 'full'}
                             })

        pool_node = Node(graph, 'pool')
        Pooling.infer(pool_node)
        res_shape = graph.node['node_2']['shape']
        self.assertIsNone(res_shape)

    def test_pooling_infer_wrong_input_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'pool'),
                             ('pool', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1, 1])},
                             'pool': {'window': np.array([1, 1, 5, 5]), 'stride': np.array([1, 1, 2, 2]),
                                      'pad': np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
                                      'pad_spatial_shape': np.array([[1, 1], [1, 1]]),
                                      'pool_method': 'avg', 'exclude_pad': False, 'global_pool': False,
                                      'output_spatial_shape': None, 'output_shape': None,
                                      'kernel_spatial': np.array([3, 3]), 'spatial_dims': np.array([2, 3]),
                                      'channel_dims': np.array([1]), 'batch_dims': np.array([0]),
                                      'pooling_convention': 'full'}
                             })

        pool_node = Node(graph, 'pool')

        with self.assertRaises(Error):
            Pooling.infer(pool_node)

    def test_pooling_infer_with_dilations(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'pool'),
                             ('pool', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256])},
                             'pool': {'window': np.array([1, 1, 2, 2]), 'stride': np.array([1, 1, 2, 2]),
                                      'pad': np.array([[0, 0], [0, 0], [0, 0], [1, 1]]),
                                      'pad_spatial_shape': np.array([[0, 0], [1, 1]]),
                                      'pool_method': 'max', 'exclude_pad': False, 'global_pool': False,
                                      'output_spatial_shape': None, 'output_shape': None,
                                      'kernel_spatial': np.array([2, 2]), 'spatial_dims': np.array([2, 3]),
                                      'channel_dims': np.array([1]), 'batch_dims': np.array([0]),
                                      'pooling_convention': 'full', 'dilation': np.array([1, 1, 2, 2]),
                                      'auto_pad': 'valid'}
                             })

        pool_node = Node(graph, 'pool')

        Pooling.infer(pool_node)
        exp_shape = np.array([1, 3, 127, 127])
        res_shape = graph.node['node_2']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
