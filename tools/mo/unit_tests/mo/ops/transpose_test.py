# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
import pytest
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, strict_compare_tensors, \
    dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.transpose import Transpose
from unit_tests.utils.graph import valued_const_with_data, result, regular_op_with_empty_data, connect, \
    build_graph, shaped_parameter

input_shape = np.array([1, 3, 224, 224])


class TestTransposeOp():
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

    @pytest.mark.parametrize("order",[list(order) for order in list(itertools.permutations(np.arange(4)))])
    def test_transpose_infer_1(self, order):
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')

        Transpose.infer(transpose_node)

        ref = [transpose_node.in_node().shape[i] for i in order]
        assert np.array_equal(transpose_node.out_node().shape, np.array(ref))

    def test_transpose_infer_2(self):
        order = None
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = True
        Transpose.infer(transpose_node)

        ref = np.array([x for x in reversed(transpose_node.in_node().shape)])
        assert np.array_equal(transpose_node.out_node().shape, ref),\
                        "Shapes are not the same: {} and {}".format(transpose_node.out_node().shape, ref)

    def test_transpose_infer_neg_1(self):
        order = np.array([0, 1, 2, 3])
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = True
        with pytest.raises(AssertionError):
            Transpose.infer(transpose_node)

    def test_transpose_infer_neg_2(self):
        order = None
        graph = self._create_graph_with_transpose(order)
        transpose_node = Node(graph, 'transpose')
        transpose_node['reverse_order'] = False
        with pytest.raises(AssertionError):
            Transpose.infer(transpose_node)


dyn = dynamic_dimension_value


class TestTransposeReverseInfer(unittest.TestCase):
    @staticmethod
    def build_and_test_reverse_inference(order, out_shape, ref_shape):
        nodes = {
            **shaped_parameter('data', None, {'reverse_infer': Parameter.reverse_infer}),
            **valued_const_with_data('order', int64_array(order)),
            **regular_op_with_empty_data('transpose', {'op': 'Transpose',
                                                       'infer': Transpose.infer,
                                                       'reverse_infer': Transpose.reverse_infer}),
            **result('res'),
        }

        edges = [
            *connect('data', '0:transpose'),
            *connect('order', '1:transpose'),
            *connect('transpose', 'res')
        ]

        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        Node(graph, 'transpose').out_port(0).data.set_shape(shape_array(out_shape))

        partial_infer(graph)
        actual_shape = Node(graph, 'data').out_port(0).data.get_shape()
        assert strict_compare_tensors(actual_shape, shape_array(ref_shape))

    def test_reverse_infer_1(self):
        self.build_and_test_reverse_inference(order=[0, 3, 1, 2],
                                              out_shape=[dyn, dyn, dyn, dyn],
                                              ref_shape=[dyn, dyn, dyn, dyn])

    def test_reverse_infer_2(self):
        self.build_and_test_reverse_inference(order=[0, 3, 1, 2],
                                              out_shape=[44, 32, 77, 1],
                                              ref_shape=[44, 77, 1, 32])

    def test_reverse_infer_3(self):
        self.build_and_test_reverse_inference(order=[0, 2, 3, 1],
                                              out_shape=[44, 32, 77, 1],
                                              ref_shape=[44, 1, 32, 77])
