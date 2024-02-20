# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, dynamic_dimension, \
    strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.pad import Pad, AttributedPad
from unit_tests.utils.graph import build_graph


class TestPadOps(unittest.TestCase):
    node_attrs = {
        'data_in': {
            'kind': 'data',
            'shape': np.array([1, 3, 100, 200]),
            'value': None,
        },
        'pads_begin': {
            'kind': 'data',
            'value': np.array([0, 0, 1, 2], dtype=np.int64),
            'shape': np.array([4], dtype=np.int64)
        },
        'pads_end': {
            'kind': 'data',
            'value': np.array([0, 0, 3, 4], dtype=np.int64),
            'shape': np.array([4], dtype=np.int64)
        },
        'pad': {
            'op': 'Pad',
            'kind': 'op',
            'pads': None,
        },
        'data_out': {
            'kind': 'data',
            'shape': None,
            'value': None,
        }
    }

    edge_attrs = [
        ('data_in', 'pad'),
        ('pad', 'data_out')
    ]

    def test_attribute_pad_no_infer(self):
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs,
            {'pad': {'pads': np.array([[0, 0], [0, 0], [1, 3], [2, 4]], dtype=np.int64)}},
            nodes_with_edges_only=True,
        )
        pad_node = Node(graph, 'pad')
        with self.assertRaisesRegex(AttributeError, ".*has no attribute 'infer'.*"):
            AttributedPad.infer(pad_node)

    def test_two_inputs(self):
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs + [('pads_begin', 'pad'), ('pads_end', 'pad')],
            nodes_with_edges_only=True,
        )
        pad_node = Node(graph, 'pad')
        Pad.infer(pad_node)
        self.assertTrue(np.array_equal(Node(graph, 'data_out').shape, np.array([1, 3, 100 + 1 + 3, 200 + 2 + 4])))

    def test_not_enough_inputs(self):
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs + [('pads_begin', 'pad')],
            nodes_with_edges_only=True,
        )
        pad_node = Node(graph, 'pad')
        with self.assertRaisesRegex(AssertionError, ".*must have 3 or 4 inputs.*"):
            Pad.infer(pad_node)

    def test_two_inputs_value_infer(self):
        in_value = np.random.rand(*self.node_attrs['data_in']['shape']).astype(np.float32)
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs + [('pads_begin', 'pad'), ('pads_end', 'pad')],
            {'data_in': {'value': in_value}},
            nodes_with_edges_only=True,
        )

        pads = np.insert(self.node_attrs['pads_end']['value'],
                         np.arange(len(self.node_attrs['pads_begin']['value'])), self.node_attrs['pads_begin']['value'])
        pads = np.reshape(pads, (len(self.node_attrs['pads_begin']['value']), 2))
        ref_value = np.pad(in_value, pads, constant_values=0, mode='constant')

        pad_node = Node(graph, 'pad')
        Pad.infer(pad_node)

        self.assertTrue(np.array_equal(Node(graph, 'data_out').shape, np.array([1, 3, 100 + 1 + 3, 200 + 2 + 4])))
        self.assertTrue(np.array_equal(Node(graph, 'data_out').value, ref_value))
        self.assertFalse(isinstance(Node(graph, 'data_out').value, np.ma.masked_array))

    def test_two_inputs_dynamic_value_infer(self):
        in_value = shape_array([dynamic_dimension_value, 3]).reshape((1, 1, 1, 2))
        graph = build_graph(
            self.node_attrs,
            self.edge_attrs + [('pads_begin', 'pad'), ('pads_end', 'pad')],
            {'data_in': {'value': in_value, 'shape': in_value.shape}},
            nodes_with_edges_only=True,
        )
        out_shape = (1, 1, 5, 8)
        mask = np.zeros(out_shape, dtype=bool)
        mask[0][0][1][2] = True
        ref_value = np.ma.masked_array(np.zeros(out_shape, dtype=np.int64), mask=mask, dtype=np.int64)
        ref_value[0][0][1][3] = 3

        pad_node = Node(graph, 'pad')
        Pad.infer(pad_node)
        output_value = Node(graph, 'data_out').value
        self.assertTrue(np.array_equal(Node(graph, 'data_out').shape, ref_value.shape))
        self.assertTrue(strict_compare_tensors(output_value, ref_value))
        self.assertTrue(isinstance(output_value, np.ma.masked_array))
        self.assertTrue(output_value[0][0][1][2] is dynamic_dimension)
