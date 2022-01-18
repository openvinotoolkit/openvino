# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.expand_dims import ExpandDims
from unit_tests.utils.graph import build_graph

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
class ExpandDimsOpDynamicDims(unittest.TestCase):
    @generate(*[(0, [1, 2, 3, dynamic_dimension_value, 224]),
                (1, [2, 1, 3, dynamic_dimension_value, 224]),
                (2, [2, 3, 1, dynamic_dimension_value, 224]),
                (3, [2, 3, dynamic_dimension_value, 1, 224]),
                (4, [2, 3, dynamic_dimension_value, 224, 1]),
                ])
    def test_expand_dims_infer(self, axis, ref_out_shape):
        graph = build_graph(nodes_attributes,
                            [('data_1', 'expand_dims'),
                             ('expand_dims', 'data_2')],
                            {'expand_dims': {'expand_axis': axis}})
        Node(graph, 'data_1').shape = shape_array([2, 3, dynamic_dimension_value, 224])
        expand_dims_node = Node(graph, 'expand_dims')

        ExpandDims.infer(expand_dims_node)

        self.assertTrue(strict_compare_tensors(expand_dims_node.out_node().shape, shape_array(ref_out_shape)))


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
