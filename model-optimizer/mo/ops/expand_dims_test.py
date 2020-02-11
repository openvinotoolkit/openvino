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
from generator import generator, generate

from mo.graph.graph import Node
from mo.ops.expand_dims import ExpandDims
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'data_1': {
        'kind': 'data',
        'shape': np.array([2, 3, 224, 224]),
        'value': None,
    },
    'expand_dims': {
        'type': 'None',
        'kind': 'op',
    },
    'data_2': {
        'kind': 'data',
        'shape': None,
        'value': None,
    }
}

@generator
class ExpandDimsOp(unittest.TestCase):
    @generate(*[(0, [1, 2, 3, 224, 224]),
                (1, [2, 1, 3, 224, 224]),
                (2, [2, 3, 1, 224, 224]),
                (3, [2, 3, 224, 1, 224]),
                (4, [2, 3, 224, 224, 1]),
                ])
    def test_expand_dims_infer(self, axis, ref_out_shape):
        graph = build_graph(nodes_attributes,
                            [('data_1', 'expand_dims'),
                             ('expand_dims', 'data_2')],
                            {'expand_dims': {'expand_axis': axis}})
        expand_dims_node = Node(graph, 'expand_dims')

        ExpandDims.infer(expand_dims_node)

        self.assertTrue(np.array_equal(expand_dims_node.out_node().shape, np.array(ref_out_shape)))


@generator
class ExpandDimsOpValueInfer(unittest.TestCase):
    @generate(*[(0, [2, 3, 224, 224], [1, 2, 3, 224, 224]),
                (1, [2, 3, 224, 224], [2, 1, 3, 224, 224]),
                (2, [2, 3, 224, 224], [2, 3, 1, 224, 224]),
                (3, [2, 3, 224, 224], [2, 3, 224, 1, 224]),
                (4, [2, 3, 224, 224], [2, 3, 224, 224, 1]),
                ])
    def test_expand_dims_infer_value(self, axis, in_shape, ref_out_shape):
        in_value = np.random.rand(*in_shape)
        graph = build_graph(nodes_attributes,
                            [('data_1', 'expand_dims'),
                             ('expand_dims', 'data_2')],
                            {'data_1': {'value': in_value},
                             'expand_dims': {'expand_axis': axis}})
        expand_dims_node = Node(graph, 'expand_dims')

        ExpandDims.infer(expand_dims_node)

        self.assertTrue(np.array_equal(expand_dims_node.out_node().shape, np.array(ref_out_shape)))
        self.assertTrue(np.array_equal(expand_dims_node.out_node().value, np.array(in_value.reshape(ref_out_shape))))
