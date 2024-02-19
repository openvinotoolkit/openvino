# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.slice_like import SliceLike
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'input': {'kind': 'op', 'op': 'Const'},
    'input_data': {'kind': 'data', 'shape': int64_array([3, 4]), 'value': np.arange(1, 13).reshape([3, 4])},
    'shape_like': {'kind': 'op', 'op': 'Const', 'shape': int64_array([2, 3]), 'value': None},
    'shape_like_data': {'kind': 'data', 'shape': int64_array([2, 3]), 'value': None},
    'slice_like': {'kind': 'op', 'op': 'slice_data'},
    'out_data': {'kind': 'data', 'shape': None, 'value': None}
}

edges = [
    ('input', 'input_data'),
    ('input_data', 'slice_like', {'in': 0}),
    ('shape_like', 'shape_like_data'),
    ('shape_like_data', 'slice_like', {'in': 1}),
    ('slice_like', 'out_data')
]


class SliceLikeTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': None}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([2, 3])
        ref_value = np.array([[1, 2, 3], [5, 6, 7]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))

    def test_2(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': (0, 1)}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([2, 3])
        ref_value = np.array([[1, 2, 3], [5, 6, 7]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))

    def test_3(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': (0,)}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([2, 4])
        ref_value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))

    def test_4(self):
        graph = build_graph(nodes_attributes, edges, {'slice_like': {'axes': (-1,)}})
        slice_like = Node(graph, 'slice_like')
        SliceLike.infer(slice_like)
        ref_shape = int64_array([3, 3])
        ref_value = np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
        res_shape = graph.node['out_data']['shape']
        res_value = graph.node['out_data']['value']
        self.assertTrue(np.array_equal(res_shape, ref_shape))
        self.assertTrue(np.array_equal(res_value, ref_value))
