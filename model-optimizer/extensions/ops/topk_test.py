"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

import numpy as np

from extensions.ops.topk import TopK
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, \
    connect, FakeAttr


class TestTopKInfer(unittest.TestCase):
    def setUp(self):
        nodes = {
            **regular_op_with_shaped_data('data', [20, 100, 4], {'type': 'Parameter', 'value': None,
                                                                 '_out_port_data_type': {0: np.float32}}),
            **valued_const_with_data('k', int64_array(10)),
            **regular_op_with_shaped_data('topk', None, {'op': 'TopK', 'type': 'TopK', 'name': 'topk', 'axis': 1}),
            'topk_d2': {'kind': 'data', 'shape': None, 'value': None},
            **result('output_1'),
            **result('output_2'),
        }

        self.graph = build_graph(nodes, [
            *connect('data', '0:topk'),
            *connect('k', '1:topk'),
            ('topk', 'topk_d', {'out': 0}),
            ('topk', 'topk_d2', {'out': 1}),
            ('topk_d', 'output_1'),
            ('topk_d2', 'output_2'),
        ], nodes_with_edges_only=True)

    def test_topk_infer_v10_opset1(self):
        self.graph.graph['cmd_params'] = FakeAttr(ir_version=10)

        topk_node = Node(self.graph, 'topk')
        topk_node['version'] = 'opset1'
        TopK.infer(topk_node)
        TopK.type_infer(topk_node)

        self.assertTrue(np.array_equal(topk_node.out_port(0).data.get_shape(), int64_array([20, 10, 4])))
        self.assertTrue(np.array_equal(topk_node.out_port(1).data.get_shape(), int64_array([20, 10, 4])))
        self.assertTrue(topk_node.out_port(0).get_data_type() == np.float32)
        self.assertTrue(topk_node.out_port(1).get_data_type() == np.int32)

    def test_topk_infer_v10_i64_opset3(self):
        self.graph.graph['cmd_params'] = FakeAttr(ir_version=10)

        topk_node = Node(self.graph, 'topk')
        topk_node['version'] = 'opset3'
        topk_node['index_element_type'] = np.int64
        TopK.infer(topk_node)
        TopK.type_infer(topk_node)

        self.assertTrue(np.array_equal(topk_node.out_port(0).data.get_shape(), int64_array([20, 10, 4])))
        self.assertTrue(np.array_equal(topk_node.out_port(1).data.get_shape(), int64_array([20, 10, 4])))
        self.assertTrue(topk_node.out_port(0).get_data_type() == np.float32)
        self.assertTrue(topk_node.out_port(1).get_data_type() == np.int64)

    def test_topk_infer_v10_i32_opset3(self):
        self.graph.graph['cmd_params'] = FakeAttr(ir_version=10)

        topk_node = Node(self.graph, 'topk')
        topk_node['version'] = 'opset3'
        topk_node['index_element_type'] = np.int32
        TopK.infer(topk_node)
        TopK.type_infer(topk_node)

        self.assertTrue(np.array_equal(topk_node.out_port(0).data.get_shape(), int64_array([20, 10, 4])))
        self.assertTrue(np.array_equal(topk_node.out_port(1).data.get_shape(), int64_array([20, 10, 4])))
        self.assertTrue(topk_node.out_port(0).get_data_type() == np.float32)
        self.assertTrue(topk_node.out_port(1).get_data_type() == np.int32)
