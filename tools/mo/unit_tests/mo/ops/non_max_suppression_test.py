# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.non_max_suppression import NonMaxSuppression
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op, regular_op_with_shaped_data, valued_const_with_data, result, connect, empty_data
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value


class TestNonMaxSuppressionInfer(unittest.TestCase):
    def setUp(self):
        nodes = {
            **regular_op_with_shaped_data('boxes', [10, 100, 4], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('scores', [10, 5, 100], {'type': 'Parameter'}),
            **valued_const_with_data('max_output_per_class', int64_array(7)),
            **regular_op('nms', {'op': 'NonMaxSuppression', 'type': 'NonMaxSuppression', 'name': 'nms'}),

            **empty_data('nms_data_0'),
            **empty_data('nms_data_1'),
            **empty_data('nms_data_2'),
            **result('output_0'),
            **result('output_1'),
            **result('output_2'),
        }

        self.graph = build_graph(nodes, [
            *connect('boxes', '0:nms'),
            *connect('scores', '1:nms'),
            *connect('max_output_per_class', '2:nms'),
            *connect('nms:0', 'nms_data_0', front_phase=True),      # Use this WA for correct creating operation
            *connect('nms_data_0', 'output_0', front_phase=True),   # with multiple outputs
        ], nodes_with_edges_only=True)

        self.graph_nms_5_2_outs = build_graph(nodes, [
            *connect('boxes', '0:nms'),
            *connect('scores', '1:nms'),
            *connect('max_output_per_class', '2:nms'),
            *connect('nms:0', 'nms_data_0', front_phase=True),      # Use this WA for correct creating operation
            *connect('nms_data_0', 'output_0', front_phase=True),   # with multiple outputs
            *connect('nms:1', 'nms_data_1', front_phase=True),
            *connect('nms_data_1', 'output_1', front_phase=True),
        ], nodes_with_edges_only=True)

        self.graph_nms_5_3_outs = build_graph(nodes, [
            *connect('boxes', '0:nms'),
            *connect('scores', '1:nms'),
            *connect('max_output_per_class', '2:nms'),
            *connect('nms:0', 'nms_data_0', front_phase=True),      # Use this WA for correct creating operation
            *connect('nms_data_0', 'output_0', front_phase=True),   # with multiple outputs
            *connect('nms:1', 'nms_data_1', front_phase=True),
            *connect('nms_data_1', 'output_1', front_phase=True),
            *connect('nms:2', 'nms_data_2', front_phase=True),
            *connect('nms_data_2', 'output_2', front_phase=True),
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

    def test_nms_infer_i32_opset5_1_out(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset5'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)

    def test_nms_infer_i64_opset5_1_out(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset5'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)

    def test_nms_infer_i32_opset5_2_outs(self):
        nms_node = Node(self.graph_nms_5_2_outs, 'nms')
        nms_node['version'] = 'opset5'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)

    def test_nms_infer_i64_opset5_2_outs(self):
        nms_node = Node(self.graph_nms_5_2_outs, 'nms')
        nms_node['version'] = 'opset5'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)

    def test_nms_infer_i32_opset5_3_outs(self):
        nms_node = Node(self.graph_nms_5_3_outs, 'nms')
        nms_node['version'] = 'opset5'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(2).data.get_shape(), [1]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)
        self.assertTrue(nms_node.out_port(2).get_data_type() == np.int64)

    def test_nms_infer_i64_opset5_3_outs(self):
        nms_node = Node(self.graph_nms_5_3_outs, 'nms')
        nms_node['version'] = 'opset5'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(2).data.get_shape(), [1]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)
        self.assertTrue(nms_node.out_port(2).get_data_type() == np.int64)

    def test_nms_infer_i32_opset9_1_out(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset9'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)

    def test_nms_infer_i64_opset9_1_out(self):
        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset9'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)

    def test_nms_infer_i32_opset9_2_outs(self):
        nms_node = Node(self.graph_nms_5_2_outs, 'nms')
        nms_node['version'] = 'opset9'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)

    def test_nms_infer_i64_opset9_2_outs(self):
        nms_node = Node(self.graph_nms_5_2_outs, 'nms')
        nms_node['version'] = 'opset9'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)

    def test_nms_infer_i32_opset9_3_outs(self):
        nms_node = Node(self.graph_nms_5_3_outs, 'nms')
        nms_node['version'] = 'opset9'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(2).data.get_shape(), [1]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)
        self.assertTrue(nms_node.out_port(2).get_data_type() == np.int64)

    def test_nms_infer_i64_opset9_3_outs(self):
        nms_node = Node(self.graph_nms_5_3_outs, 'nms')
        nms_node['version'] = 'opset9'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(1).data.get_shape(),
                                       shape_array([dynamic_dimension_value, 3])))
        self.assertTrue(np.array_equal(nms_node.out_port(2).data.get_shape(), [1]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)
        self.assertTrue(nms_node.out_port(1).get_data_type() == np.float32)
        self.assertTrue(nms_node.out_port(2).get_data_type() == np.int64)
