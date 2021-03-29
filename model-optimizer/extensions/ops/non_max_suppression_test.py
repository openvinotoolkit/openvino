# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.ops.non_max_suppression import NonMaxSuppression
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect


class TestNonMaxSuppressionInfer(unittest.TestCase):
    def setUp(self):
        nodes = {
            **regular_op_with_shaped_data('boxes', [10, 100, 4], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('scores', [10, 5, 100], {'type': 'Parameter'}),
            **valued_const_with_data('max_output_per_class', int64_array(7)),
            **regular_op_with_shaped_data('nms', None, {'op': 'NonMaxSuppression', 'type': 'NonMaxSuppression',
                                                        'name': 'nms'}),
            **result('output'),
        }

        self.graph = build_graph(nodes, [
            *connect('boxes', '0:nms'),
            *connect('scores', '1:nms'),
            *connect('max_output_per_class', '2:nms'),
            *connect('nms', 'output'),
        ], nodes_with_edges_only=True)

    def test_nms_infer_opset1(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset1'
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [100, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)

    def test_nms_infer_i64_opset3(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset3'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [100, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)

    def test_nms_infer_i32_opset3(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset3'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [100, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)

    def test_nms_infer_i32_opset4(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset4'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [10 * 5 * 7, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)

    def test_nms_infer_i64_opset4(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset4'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [10 * 5 * 7, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)
