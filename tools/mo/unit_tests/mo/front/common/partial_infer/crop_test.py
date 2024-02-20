# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.crop import crop_infer
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'data'},
                    'node_2': {'value': None, 'kind': 'data'},
                    'crop_1': {'op': 'Crop', 'kind': 'op'},
                    'node_3': {'value': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }


class TestCropInfer(unittest.TestCase):
    def test_crop_infer_ideal(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'crop_1'),
                             ('node_2', 'crop_1'),
                             ('crop_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 2, 500, 500])},
                             'node_2': {'shape': np.array([1, 2, 256, 256])},
                             'crop_1': {'axis': 2, 'offset': [0, 0], 'dim': None}
                             })

        crop_node = Node(graph, 'crop_1')

        crop_infer(crop_node)
        exp_shape = np.array([1, 2, 256, 256])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(crop_node.axis, [2, 3])
        self.assertEqual(crop_node.offset, [0, 0])
        self.assertEqual(crop_node.dim, [256, 256])

    def test_crop_infer_negative_axis(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'crop_1'),
                             ('node_2', 'crop_1'),
                             ('crop_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 2, 500, 500])},
                             'node_2': {'shape': np.array([1, 2, 256, 256])},
                             'crop_1': {'axis': -1, 'offset': [0, 0], 'dim': None}
                             })

        crop_node = Node(graph, 'crop_1')

        crop_infer(crop_node)
        exp_shape = np.array([1, 2, 500, 256])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(crop_node.axis, [3])
        self.assertEqual(crop_node.offset, [0])
        self.assertEqual(crop_node.dim, [256])

    def test_crop_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'crop_1'),
                             ('node_2', 'crop_1'),
                             ('crop_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 2, 500, 500])},
                             'node_2': {'shape': None},
                             'crop_1': {'axis': 2, 'offset': [0, 0], 'dim': None}
                             })

        crop_node = Node(graph, 'crop_1')

        crop_infer(crop_node)
        self.assertIsNone(graph.node['node_3']['shape'])

    def test_crop_infer_one_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'crop_1'),
                             ('crop_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 2, 500, 500])},
                             'crop_1': {'axis': 2, 'offset': [0], 'dim': None}
                             })

        crop_node = Node(graph, 'crop_1')

        crop_infer(crop_node)
        self.assertIsNone(graph.node['node_3']['shape'])

    def test_crop_infer_out_offset(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'crop_1'),
                             ('node_2', 'crop_1'),
                             ('crop_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 2, 500, 500])},
                             'node_2': {'shape': np.array([1, 2, 256, 256])},
                             'crop_1': {'axis': 2, 'offset': [300], 'dim': None}
                             })

        crop_node = Node(graph, 'crop_1')

        crop_infer(crop_node)
        self.assertIsNone(graph.node['node_3']['shape'])
