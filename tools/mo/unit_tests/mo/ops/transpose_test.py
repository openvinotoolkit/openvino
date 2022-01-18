# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np
from generator import generator, generate

from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

input_shape = np.array([1, 3, 224, 224])


@generator
class TestTransposeOp(unittest.TestCase):
    nodes_attributes = {
        'parameter': {
            'kind': 'op',
            'op': 'Parameter',
            'shape': input_shape
        },
        'data_1': {
            'kind': 'data',
            'shape': input_shape,
            'value': None
        },
        'order_const': {
            'kind': 'op',
            'op': 'Const',
            'shape': np.array([4])
        },
        'order_data': {
            'kind': 'data',
            'shape': np.array([4])
        },
        'transpose': {
            'type': 'Transpose',
            'op': 'Transpose',
            'reverse_order': False,
            'kind': 'op',
        },
        'data_2': {
            'kind': 'data',
            'shape': None,
            'value': None
        }
    }

    def _create_graph_with_transpose(self, order):
        if order is None:
            graph = build_graph(self.nodes_attributes,
                                [('parameter', 'data_1'),
                                 ('data_1', 'transpose'),
                                 ('transpose', 'data_2')])
        else:
            graph = build_graph(self.nodes_attributes,
                                [('parameter', 'data_1'),
                                 ('data_1', 'transpose'),
                                 ('order_const', 'order_data'),
                                 ('order_data', 'transpose'),
                                 ('transpose', 'data_2')],
                                {'order_data': {'value': order}})
        graph.graph['layout'] = 'NCHW'
        return graph

    @generate(*[list(order) for order in list(itertools.permutations(np.arange(4)))])
    def test_transpose_infer_1(self, order):
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')

        Transpose.infer(transpose_node)

        ref = [transpose_node.in_node().shape[i] for i in order]
        self.assertTrue(np.array_equal(transpose_node.out_node().shape, np.array(ref)))

    def test_transpose_infer_2(self):
        order = None
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = True
        Transpose.infer(transpose_node)

        ref = np.array([x for x in reversed(transpose_node.in_node().shape)])
        self.assertTrue(np.array_equal(transpose_node.out_node().shape, ref),
                        "Shapes are not the same: {} and {}".format(transpose_node.out_node().shape, ref))

    def test_transpose_infer_neg_1(self):
        order = np.array([0, 1, 2, 3])
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = True
        self.assertRaises(AssertionError, Transpose.infer, transpose_node)

    def test_transpose_infer_neg_2(self):
        order = None
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = False
        self.assertRaises(AssertionError, Transpose.infer, transpose_node)
