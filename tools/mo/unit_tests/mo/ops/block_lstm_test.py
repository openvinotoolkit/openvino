# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import strict_compare_tensors, shape_array, dynamic_dimension
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.BlockLSTM import BlockLSTM
from unit_tests.utils.graph import build_graph, result, connect, regular_op_with_shaped_data, regular_op_with_empty_data


class TestBlockLSTM(unittest.TestCase):

    def run_test(self, time_len, batch_size, num_inputs, hidden_size, ref_hidden_states_shape, ref_cell_states_shape):
        nodes = {
            **regular_op_with_shaped_data('x', [time_len, batch_size, num_inputs], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('weights', [num_inputs, hidden_size * 4], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('bias', [hidden_size * 4], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('init_hidden_state', [hidden_size * 4], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('init_cell_state', [hidden_size * 4], {'type': 'Parameter'}),
            **regular_op_with_empty_data('block_lstm', {'op': 'BlockLSTM'}),
            **result('hidden_states'),
            **result('cell_states'),
        }

        edges = [
            *connect('x', '0:block_lstm'),
            *connect('weights', '1:block_lstm'),
            *connect('bias', '2:block_lstm'),
            *connect('init_hidden_state', '3:block_lstm'),
            *connect('init_cell_state', '4:block_lstm'),
            *connect('block_lstm:0', 'hidden_states'),
            *connect('block_lstm:1', 'cell_states')
        ]

        graph = build_graph(nodes, edges)
        node = Node(graph, 'block_lstm')
        BlockLSTM.infer(node)
        hidden_states_shape = node.out_port(0).data.get_shape()
        cell_states_shape = node.out_port(1).data.get_shape()
        self.assertTrue(strict_compare_tensors(hidden_states_shape, ref_hidden_states_shape))
        self.assertTrue(strict_compare_tensors(cell_states_shape, ref_cell_states_shape))

    def test_block_lstm_basic_infer(self):
        self.run_test(time_len=4, batch_size=2, num_inputs=40, hidden_size=60,
                      ref_hidden_states_shape=shape_array([4, 2, 60]), ref_cell_states_shape=shape_array([4, 2, 60]))

    def test_block_lstm_dynamic_infer(self):
        self.run_test(time_len=dynamic_dimension, batch_size=dynamic_dimension, num_inputs=40, hidden_size=60,
                      ref_hidden_states_shape=shape_array([dynamic_dimension, dynamic_dimension, 60]),
                      ref_cell_states_shape=shape_array([dynamic_dimension, dynamic_dimension, 60]))

    def test_failed_three_outputs(self):
        nodes = {
            **regular_op_with_shaped_data('x', [30, 3, 70], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('weights', [70, 120], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('bias', [120], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('init_hidden_state', [120], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('init_cell_state', [120], {'type': 'Parameter'}),
            **regular_op_with_empty_data('block_lstm', {'op': 'BlockLSTM'}),
            **result('hidden_states'),
            **result('cell_states'),
            **result('f'),
        }

        edges = [
            *connect('x', '0:block_lstm'),
            *connect('weights', '1:block_lstm'),
            *connect('bias', '2:block_lstm'),
            *connect('init_hidden_state', '3:block_lstm'),
            *connect('init_cell_state', '4:block_lstm'),
            *connect('block_lstm:0', 'hidden_states'),
            *connect('block_lstm:1', 'cell_states'),
            *connect('block_lstm:2', 'f')
        ]

        graph = build_graph(nodes, edges)
        node = Node(graph, 'block_lstm')
        with self.assertRaisesRegex(AssertionError, 'Internal Model Optimizer Error or unsupported BlockLSTM*'):
            BlockLSTM.infer(node)
