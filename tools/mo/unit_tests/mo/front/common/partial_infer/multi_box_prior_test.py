# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.multi_box_prior import multi_box_prior_infer_mxnet
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'data'},
                    'node_2': {'value': None, 'kind': 'data'},
                    'prior_box_1': {'type': 'PriorBox', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'data'}
                    }


class TestMultiBoxPriorInfer(unittest.TestCase):
    def test_prior_box_infer_ideal(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'prior_box_1'),
                             ('node_2', 'prior_box_1'),
                             ('prior_box_1', 'node_3')],
                            {'node_1': {'shape': np.array([1, 1024, 19, 19])},
                             'node_2': {'shape': np.array([1, 3, 300, 300])},
                             'prior_box_1': {'aspect_ratio': [1.0, 2.0, 0.5, 3.0, 0.333333333333],
                                             'min_size': [0.2, 0.272],
                                             'max_size': '', 'offset': 0.5, 'step': 0.2, 'sizes': [0.2, 0.272]},
                             'node_3': {'shape': np.array([1, 2, 3])},
                             })

        multi_box_prior_node = Node(graph, 'prior_box_1')

        multi_box_prior_infer_mxnet(multi_box_prior_node)
        exp_shape = np.array([1, 2, 8664])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(multi_box_prior_node.min_size, [0.2, 0.272])
        self.assertEqual(multi_box_prior_node.max_size, '')
        self.assertEqual(multi_box_prior_node.aspect_ratio, [1.0, 2.0, 0.5, 3.0, 0.333333333333])
        self.assertEqual(round(multi_box_prior_node.step, 1), 0.2)
        self.assertEqual(round(multi_box_prior_node.offset, 1), 0.5)
