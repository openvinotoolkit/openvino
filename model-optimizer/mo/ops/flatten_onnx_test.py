"""
 Copyright (c) 2018 Intel Corporation

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
from generator import generator, generate

from mo.graph.graph import Node
from mo.ops.flatten_onnx import FlattenONNX
from mo.utils.unittest.graph import build_graph


@generator
class TestFlattenONNXOp(unittest.TestCase):
    # There are tests for InnerProduct.infer in mo/front/common/partial_infer/inner_product_test.py
    nodes_attributes = {
        'data_1': {
            'kind': 'data',
            'shape': np.array([1, 3, 224, 224])
        },
        'flatten': {
            'type': 'Reshape',
            'axis': None,
            'kind': 'op',
        },
        'data_2': {
            'kind': 'data',
            'shape': None,
        }
    }

    def _create_graph_with_flatten(self, axis):
        graph = build_graph(self.nodes_attributes,
                            [('data_1', 'flatten'),
                             ('flatten', 'data_2')],
                            {'flatten': {'axis': axis}})
        return graph

    @generate(*[(0, [1, 3 * 224 * 224]),
                (1, [1, 3 * 224 * 224]),
                (2, [3, 224 * 224]),
                (3, [3 * 224, 224]),
                (4, [3 * 224 * 224, 1]),
                ])
    def test_flatten_infer_1(self, axis, ref):
        graph = self._create_graph_with_flatten(axis)
        flatten_node = Node(graph, 'flatten')

        FlattenONNX.infer(flatten_node)

        self.assertTrue(np.array_equal(flatten_node.out_node().shape, np.array(ref)))
