# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.activation_ops import Elu, SoftPlus, Mish, Swish, SoftSign
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestActivationOp(unittest.TestCase):
    nodes_attributes = {
        'node_1': {
            'shape': np.array([4]),
            'value': None
        },
        'activation_node': {
            'op': 'Activation',
            'kind': 'op',
            'operation': None
        },
        'node_3': {
            'shape': None
        }
    }

    def test_activation_elu_infer(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'activation_node'),
                                ('activation_node', 'node_3')
                            ],
                            {
                                'node_1': {
                                    'value': np.array([6, -4, -2, -1])
                                },
                                'activation_node': {
                                    'operation': 'elu',
                                    'alpha': 1.0,
                                },
                                'node_3': {
                                    'value': None
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        activation_node = Node(graph, 'activation_node')
        Elu.infer(activation_node)
        exp_shape = np.array([4])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([6., -0.98168436, -0.86466472, -0.63212056])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)

    def test_activation_softplus_infer(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'activation_node'),
                                ('activation_node', 'node_3')
                            ],
                            {
                                'node_1': {
                                    'value': np.array([-1.0, 0.0, 1.0, 20.0])
                                },
                                'activation_node': {
                                    'op': 'SoftPlus',
                                    'operation': SoftPlus.operation,
                                },
                                'node_3': {
                                    'value': None
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        activation_node = Node(graph, 'activation_node')
        SoftPlus.infer(activation_node)
        exp_shape = np.array([4])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([0.3132617, 0.6931472, 1.3132617, 20.0])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)

    def test_activation_mish_infer(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'activation_node'),
                                ('activation_node', 'node_3')
                            ],
                            {
                                'node_1': {
                                    'value': np.array([-1.0, 0.0, 1.0, 20.0])
                                },
                                'activation_node': {
                                    'op': 'Mish',
                                    'operation': Mish.operation,
                                },
                                'node_3': {
                                    'value': None
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        activation_node = Node(graph, 'activation_node')
        Mish.infer(activation_node)
        exp_shape = np.array([4])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([-0.30340146, 0.0, 0.8650984, 20.0])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)

    def test_activation_swish_infer(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'activation_node'),
                                ('activation_node', 'node_3')
                            ],
                            {
                                'node_1': {
                                    'value': np.array([-1.0, 0.0, 1.0, 20.0])
                                },
                                'activation_node': {
                                    'op': 'Swish',
                                },
                                'node_3': {
                                    'value': None
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        activation_node = Node(graph, 'activation_node')
        Swish.infer(activation_node)
        exp_shape = np.array([4])
        res_shape = graph.node['node_3']['shape']
        res_value = graph.node['node_3']['value']
        exp_value = np.array([-0.26894142, 0.0, 0.73105858, 19.99999996])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)

    def test_activation_softsign_infer(self):
        graph = build_graph(self.nodes_attributes,
                            edges=[
                                ('node_1', 'activation_node'),
                                ('activation_node', 'node_3')
                            ],
                            update_attributes={
                                'node_1': {
                                    'value': np.array([1.0, -1.0, 3.5, -5.8])
                                },
                                'activation_node': {
                                    'op': 'SoftSign',
                                    'operation': SoftSign.operation
                                },
                                'node_3': {
                                    'value': None
                                }
                            })
        graph.graph['layout'] = 'NCHW'
        activation_node = Node(graph, 'activation_node')
        SoftSign.infer(activation_node)
        exp_shape = np.array([4])
        res_shape = graph.nodes['node_3']['shape']
        res_value = graph.nodes['node_3']['value']
        exp_value = np.array([0.5, -0.5, 0.7777777777777, -0.85294117647])
        for i, value in enumerate(exp_shape):
            self.assertEqual(res_shape[i], value)
        for i, value in enumerate(exp_value):
            self.assertAlmostEqual(res_value[i], value)
