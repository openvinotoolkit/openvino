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

from extensions.ops.non_max_suppression import NonMaxSuppression
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, \
    connect, FakeAttr


class TestNonMaxSuppressionInfer(unittest.TestCase):
    def setUp(self):
        nodes = {
            **regular_op_with_shaped_data('boxes', [10, 100, 4], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('scores', [10, 5, 100], {'type': 'Parameter'}),
            **valued_const_with_data('max_output_per_class', int64_array(10)),
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

    def test_nms_infer_v10_opset1(self):
        self.graph.graph['cmd_params'] = FakeAttr(ir_version=10)

        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset1'
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [100, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)

    def test_nms_infer_v10_i64_opset3(self):
        self.graph.graph['cmd_params'] = FakeAttr(ir_version=10)

        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset3'
        nms_node['output_type'] = np.int64
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [100, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int64)

    def test_nms_infer_v10_i32_opset3(self):
        self.graph.graph['cmd_params'] = FakeAttr(ir_version=10)

        nms_node = Node(self.graph, 'nms')
        nms_node['version'] = 'opset3'
        nms_node['output_type'] = np.int32
        NonMaxSuppression.infer(nms_node)
        NonMaxSuppression.type_infer(nms_node)

        self.assertTrue(np.array_equal(nms_node.out_port(0).data.get_shape(), [100, 3]))
        self.assertTrue(nms_node.out_port(0).get_data_type() == np.int32)
