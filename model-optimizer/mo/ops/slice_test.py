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
from generator import generator

from mo.graph.graph import Node
from mo.ops.slice import Slice
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'data_1': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'begin': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'size': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'slice': {
        'op': 'Slice',
        'axis': None,
        'start': None,
        'end': None,
        'kind': 'op',
    },
    'data_2': {
        'kind': 'data',
        'shape': None,
        'value': None,
    }
}


@generator
class TestSliceOp(unittest.TestCase):
    def test_slice_infer_constant(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'slice'),
                             ('begin', 'slice'),
                             ('size', 'slice'),
                             ('slice', 'data_2')],
                            {'data_1': {'shape': np.array([4]), 'value': np.array([1, 3, 224, 224])},
                             'slice': {'start': np.array([1]), 'end': np.array([2])},
                             'size': {'value': np.array([1])},
                             'begin': {'value': np.array([1])}})

        slice_node = Node(graph, 'slice')
        Slice.infer(slice_node)

        self.assertTrue(np.array_equal(slice_node.out_node().value, np.array([3])))
        self.assertTrue(np.array_equal(slice_node.out_node().shape, np.array([1])))
        self.assertTrue(np.array_equal(slice_node['slices'], np.array([slice(1, 2, 1)])))

    def test_slice_infer_non_constant(self):
        # Testing non-constant path case (when value in input is None)
        # with multiply params
        graph = build_graph(nodes_attributes,
                            [('data_1', 'slice'),
                             ('begin', 'slice'),
                             ('size', 'slice'),
                             ('slice', 'data_2')],
                            {'data_1': {'shape': np.array([4, 5, 6])},
                             'slice': {'start': np.array([1, 2]),
                                       'end': np.array([4, 3])},
                             'size': {'value': np.array([3, 1])},
                             'begin': {'value': np.array([1, 2])}})

        slice_node = Node(graph, 'slice')

        Slice.infer(slice_node)
        self.assertTrue(np.array_equal(slice_node.out_node().value, None))
        self.assertTrue(np.array_equal(slice_node.out_node().shape, np.array([3, 1, 6])))
        self.assertTrue(np.array_equal(slice_node['slices'], np.array([slice(1, 4, 1), slice(2, 3, 1), slice(0, 6, 1)])))

    def test_slice_infer_multiply_params(self):
        # Test case when size[i] == -1 (that means all
        # remaining elements in dimension i are included in the slice)
        graph = build_graph(nodes_attributes,
                            [('data_1', 'slice'),
                             ('begin', 'slice'),
                             ('size', 'slice'),
                             ('slice', 'data_2')],
                            {'data_1': {'shape': np.array([4, 5, 6])},
                             'slice': {'start': np.array([1, 2]),
                                       'end': np.array([4, 1])},
                             'size': {'value': np.array([3, -1])},
                             'begin': {'value': np.array([1, 2])}})

        slice_node = Node(graph, 'slice')

        Slice.infer(slice_node)
        self.assertTrue(np.array_equal(slice_node.out_node().value, None))
        self.assertTrue(np.array_equal(slice_node.out_node().shape, np.array([3, 3, 6])))
        self.assertTrue(np.array_equal(slice_node['slices'], np.array([slice(1, 4, 1), slice(2, 5, 1), slice(0, 6, 1)])))
