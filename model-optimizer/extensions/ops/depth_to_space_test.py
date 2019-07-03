"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.ops.depth_to_space import DepthToSpaceOp
from mo.graph.graph import Node
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph

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
    def test_tf_depth_to_space_infer(self):
        graph = build_graph(nodes, edges)
        dts_node = Node(graph, 'DtS')
        DepthToSpaceOp.depth_to_space_infer(dts_node)
        exp_shape = np.array([1, 2048, 1152, 64])
        res_shape = graph.node['out_data_node']['shape']
        self.assertTrue(np.array_equal(exp_shape, res_shape))

    def test_tf_depth_to_space_infer_error(self):
        graph = build_graph(nodes, edges)
        graph.node['in_data_node']['shape'] = np.array([1024, 576, 256])
        dts_node = Node(graph, 'DtS')
        self.assertRaises(Error, DepthToSpaceOp.depth_to_space_infer(dts_node))

    def test_tf_depth_to_space_infer_error_1(self):
        graph = build_graph(nodes, edges)
        graph.node['in_data_node']['shape'] = np.array([1, 1024, 576, 255])
        dts_node = Node(graph, 'DtS')
        self.assertRaises(Error, DepthToSpaceOp.depth_to_space_infer(dts_node))
