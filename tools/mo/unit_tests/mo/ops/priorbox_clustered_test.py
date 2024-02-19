# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.priorbox_clustered import PriorBoxClusteredOp
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'data'},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'data'},
                    'pbc': {'type': 'PriorBoxClustered', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }


class TestPriorBoxClusteredPartialInfer(unittest.TestCase):
    def test_caffe_priorboxclustered_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'pbc'),
                                ('node_2', 'pbc'),
                                ('pbc', 'node_3'),
                                ('node_3', 'op_output')
                             ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 384, 19, 19])},
                                'node_2': {'shape': np.array([1, 3, 300, 300])},
                                'pbc': {'flip': 0, 'clip': 0, 'variance': [0.1, 0.1, 0.2, 0.2],
                                        'step': 0, 'offset': 0.5, 'width': [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                        'height': [2., 2., 2., 2., 2., 2., 2., 2., 2.]}
                            })
        graph.graph['layout'] = 'NCHW'

        pbc_node = Node(graph, 'pbc')
        PriorBoxClusteredOp.priorbox_clustered_infer(pbc_node)
        exp_shape = np.array([1, 2, 12996])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_priorboxclustered_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'pbc'),
                                ('node_2', 'pbc'),
                                ('pbc', 'node_3'),
                                ('node_3', 'op_output')
                             ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 19, 19, 384])},
                                'node_2': {'shape': np.array([1, 300, 300, 3])},
                                'pbc': {'flip': 0, 'clip': 0, 'variance': [0.1, 0.1, 0.2, 0.2],
                                        'step': 0, 'offset': 0.5, 'width': [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                        'height': [2., 2., 2., 2., 2., 2., 2., 2., 2.]}
                            })
        graph.graph['layout'] = 'NHWC'

        pbc_node = Node(graph, 'pbc')
        PriorBoxClusteredOp.priorbox_clustered_infer(pbc_node)
        exp_shape = np.array([1, 2, 12996])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
