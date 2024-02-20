# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.argmax import arg_ops_infer
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {
                    'op_input': {'kind': 'op', 'op': 'Parameter'},
                    'node_1': {'kind': 'data'},
                    'argmax': {'op': 'ArgMax', 'kind': 'op'},
                    'node_3': {'kind': 'data', 'value': None},
                    'op_output': {'kind': 'op', 'op': 'Result'}
                    }


class TestArgMaxOp(unittest.TestCase):
    def test_caffe_argmax_axis(self):
        graph = build_graph(nodes_attributes,
                            [
                             ('op_input', 'node_1'),
                             ('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1025, 2049])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100,
                                 'axis': 2
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        arg_ops_infer(argmax_node)
        exp_shape = np.array([1, 3, 100, 2049])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_axis_negative(self):
        graph = build_graph(nodes_attributes,
                            [
                             ('op_input', 'node_1'),
                             ('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1025, 2049])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100,
                                 'axis': -1
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        arg_ops_infer(argmax_node)
        exp_shape = np.array([1, 3, 1025, 100])
        res_shape = graph.node['node_3']['shape']
        self.assertEqual(argmax_node.axis, 3)
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_no_axis(self):
        graph = build_graph(nodes_attributes,
                            [
                             ('op_input', 'node_1'),
                             ('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1025, 2049])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        arg_ops_infer(argmax_node)
        exp_shape = np.array([1, 2, 100, 1])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_extend_shape(self):
        graph = build_graph(nodes_attributes,
                            [
                             ('op_input', 'node_1'),
                             ('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3])},
                             'argmax': {
                                 'out_max_val': True,
                                 'top_k': 100
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        arg_ops_infer(argmax_node)
        exp_shape = np.array([1, 2, 100])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_argmax_out_max_val_false(self):
        graph = build_graph(nodes_attributes,
                            [
                             ('op_input', 'node_1'),
                             ('node_1', 'argmax'),
                             ('argmax', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3])},
                             'argmax': {
                                 'out_max_val': False,
                                 'top_k': 100
                             }
                             })

        argmax_node = Node(graph, 'argmax')
        arg_ops_infer(argmax_node)
        exp_shape = np.array([1, 1, 100])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
