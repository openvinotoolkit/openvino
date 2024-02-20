# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.BlockLSTMtoLSTMSequence import BlockLSTMtoLSTMSequenceSingleFirstOutput
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import regular_op_with_empty_data, result, build_graph, connect, \
    valued_const_with_data

nodes_attributes = {
    # hidden_size = 2
    **regular_op_with_empty_data('x', {'op': 'Parameter', 'type': 'Parameter'}),
    **valued_const_with_data('weights', np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                                                  [9, 10, 11, 12, 13, 14, 15, 16]], dtype=np.float32)),
    **valued_const_with_data('bias', np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=np.float32)),
    **regular_op_with_empty_data('h_init_state', {'op': 'Parameter', 'type': 'Parameter'}),
    **regular_op_with_empty_data('c_init_state', {'op': 'Parameter', 'type': 'Parameter'}),
    **regular_op_with_empty_data('block_lstm', {'op': 'BlockLSTM', 'type': None, 'forget_bias': 1}),
    **result('result'),

    **valued_const_with_data('weights_normalized', np.array(
        [[5, 13], [6, 14], [1, 9], [2, 10], [3, 11],
         [4, 12], [7, 15], [8, 16]],
        dtype=np.float32)),
    **valued_const_with_data('bias_normalized', np.array([11, 13, 2, 4, 6, 8, 14, 16], dtype=np.float32)),
}


class BlockLSTMtoLSTMSequenceSingleFirstOutputTest(unittest.TestCase):

    def test_tf_block_lstm_to_lstm_seq(self):
        graph = build_graph(nodes_attrs=nodes_attributes,
                            edges=[
                                *connect('x', '0:block_lstm'),
                                *connect('weights', '1:block_lstm'),
                                *connect('bias', '2:block_lstm'),
                                *connect('h_init_state', '3:block_lstm'),
                                *connect('c_init_state', '4:block_lstm'),
                                *connect('block_lstm', 'result')
                            ],
                            nodes_with_edges_only=True)
        BlockLSTMtoLSTMSequenceSingleFirstOutput().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                edges=[
                                    *connect('x', '0:block_lstm'),
                                    *connect('weights_normalized', '1:block_lstm'),
                                    *connect('bias_normalized', '2:block_lstm'),
                                    *connect('h_init_state', '4:block_lstm'),
                                    *connect('c_init_state', '5:block_lstm'),
                                    *connect('block_lstm', 'result'),
                                ],
                                update_attributes={
                                    'block_lstm': {'sequence_dim': 0, 'batch_dim': 1, 'direction': 'forward',
                                                   'hidden_size': 2, 'format': 'tf', 'type': 'RNNSequence',
                                                   'op': 'LSTM'},
                                },
                                nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
