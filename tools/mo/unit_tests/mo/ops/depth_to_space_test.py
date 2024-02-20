# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from openvino.tools.mo.ops.depth_to_space import DepthToSpaceOp
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph

nodes = {
    'in_data_node': {'value': None, 'kind': 'data', 'shape': np.array([1, 1024, 576, 256])},
    'DtS': {'op': 'DepthToSpace', 'kind': 'op', 'block_size': 2},
    'out_data_node': {'value': None, 'kind': 'data', 'shape': None}
}

edges = [
    ('in_data_node', 'DtS'),
    ('DtS', 'out_data_node')
]


class TestDepthToSpacePartialInfer(unittest.TestCase):
    def test_tf_depth_to_space_infer_nhwc(self):
        graph = build_graph(nodes, edges)
        graph.graph['layout'] = 'NHWC'
        dts_node = Node(graph, 'DtS')
        DepthToSpaceOp.infer(dts_node)
        exp_shape = np.array([1, 2048, 1152, 64])
        res_shape = graph.node['out_data_node']['shape']
        self.assertTrue(np.array_equal(exp_shape, res_shape))

    def test_tf_depth_to_space_infer_nchw(self):
        graph = build_graph(nodes, edges)
        graph.graph['layout'] = 'NCHW'
        graph.node['in_data_node']['shape'] = np.array([1, 256, 1024, 576])
        dts_node = Node(graph, 'DtS')
        DepthToSpaceOp.infer(dts_node)
        exp_shape = np.array([1, 64, 2048, 1152])
        res_shape = graph.node['out_data_node']['shape']
        self.assertTrue(np.array_equal(exp_shape, res_shape))

    def test_tf_depth_to_space_infer_error(self):
        graph = build_graph(nodes, edges)
        graph.graph['layout'] = 'NHWC'
        graph.node['in_data_node']['shape'] = np.array([1024, 576, 256])
        dts_node = Node(graph, 'DtS')
        self.assertRaises(Error, DepthToSpaceOp.infer, dts_node)

    def test_tf_depth_to_space_infer_divisibility_error_1(self):
        graph = build_graph(nodes, edges)
        graph.graph['layout'] = 'NHWC'
        graph.node['in_data_node']['shape'] = np.array([1, 1024, 576, 255])
        dts_node = Node(graph, 'DtS')
        self.assertRaises(Error, DepthToSpaceOp.infer, dts_node)

    def test_tf_depth_to_space_infer_divisibility_error_2(self):
        graph = build_graph(nodes, edges)
        graph.graph['layout'] = 'NCHW'
        graph.node['in_data_node']['shape'] = np.array([1, 255, 1024, 576])
        dts_node = Node(graph, 'DtS')
        self.assertRaises(Error, DepthToSpaceOp.infer, dts_node)

