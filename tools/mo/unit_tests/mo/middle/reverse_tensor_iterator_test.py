# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.reverse_tensor_iterator import ReverseTensorIteratorLSTM
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, connect, \
    valued_const_with_data, regular_op_with_empty_data, result

nodes = {
    **regular_op_with_shaped_data('parameter', [1, 3, 227, 227],
                                  {'type': 'Parameter', 'op': 'Parameter', 'shape': [1, 3, 227, 227]}),
    **valued_const_with_data('seq_len', np.array([227])),
    **regular_op_with_empty_data('shapeof', {'type': 'ShapeOf', 'op': 'ShapeOf'}),
    **valued_const_with_data('gather_axis', np.array([0])),
    **valued_const_with_data('gather_batch_ind', np.array([0])),
    **valued_const_with_data('gather_seq_ind', np.array([2])),
    **regular_op_with_empty_data('gather_batch', {'type': 'Gather', 'op': 'Gather'}),
    **regular_op_with_empty_data('gather_seq', {'type': 'Gather', 'op': 'Gather'}),
    **regular_op_with_empty_data('broadcast', {'type': 'Broadcast', 'op': 'Broadcast'}),
    **regular_op_with_shaped_data('direct_reverse', [1, 3, 227, 227], {'type': 'ReverseSequence',
                                                                       'op': 'ReverseSequence',
                                                                       'seq_axis': 2, 'batch_axis': 0}),
    **regular_op_with_empty_data('init_hidden', {'type': 'Init', 'op': 'Init'}),

    **regular_op_with_shaped_data('ti', [1, 2, 34, 56], {'type': 'TensorIterator', 'op': 'TensorIterator',
                                                         'output_port_map': [{'axis': 2, 'start': 0, 'end': -1,
                                                                              'stride': 1, 'external_port_id': 0}],
                                                         'input_port_map': [{'axis': 2, 'start': -1, 'end': 0,
                                                                             'stride': -1, 'external_port_id': 0}]}),
    **valued_const_with_data('inverse_seq_len', np.array([34])),
    **regular_op_with_empty_data('inverse_shapeof', {'type': 'ShapeOf', 'op': 'ShapeOf'}),
    **regular_op_with_empty_data('inverse_gather_batch', {'type': 'Gather', 'op': 'Gather'}),
    **regular_op_with_empty_data('inverse_gather_seq', {'type': 'Gather', 'op': 'Gather'}),
    **regular_op_with_empty_data('inverse_broadcast', {'type': 'Broadcast', 'op': 'Broadcast'}),
    **regular_op_with_shaped_data('inverse_reverse', [1, 2, 34, 56], {'type': 'ReverseSequence',
                                                                      'op': 'ReverseSequence',
                                                                      'seq_axis': 2, 'batch_axis': 0}),
    **regular_op_with_empty_data('some_op', {'op': 'SomeOp'}),
    **result()
}

ref_nodes = {
    **regular_op_with_shaped_data('parameter', [1, 3, 227, 227],
                                  {'type': 'Parameter', 'op': 'Parameter', 'shape': [1, 3, 227, 227]}),
    **regular_op_with_empty_data('init_hidden', {'type': 'Init', 'op': 'Init'}),
    **regular_op_with_empty_data('ti', {'type': 'TensorIterator', 'op': 'TensorIterator',
                                        'output_port_map': [{'axis': 2, 'start': -1, 'end': 0, 'stride': -1,
                                                             'external_port_id': 0}],
                                        'input_port_map': [{'axis': 2, 'start': 0, 'end': -1, 'stride': 1,
                                                            'external_port_id': 0}]}),
    **regular_op_with_empty_data('some_op', {'op': 'SomeOp'}),
    **result()
}


class ReverseTensorIteratorTest(unittest.TestCase):
    def test_ti_reverse(self):
        graph = build_graph(nodes, [*connect('parameter:0', '0:direct_reverse'),
                                    *connect('parameter:0', 'shapeof', skip_data=True),
                                    *connect('shapeof:0', '0:gather_batch'),
                                    *connect('gather_batch_ind', '1:gather_batch'),
                                    *connect('gather_axis', '2:gather_batch'),
                                    *connect('shapeof:0', '0:gather_seq', skip_data=True),
                                    *connect('gather_seq_ind', '1:gather_seq'),
                                    *connect('gather_axis', '2:gather_seq'),
                                    *connect('gather_seq', '0:broadcast'),
                                    *connect('gather_batch', '1:broadcast'),
                                    *connect('broadcast', '1:direct_reverse'),
                                    *connect('direct_reverse', '0:ti'),
                                    *connect('init_hidden', '1:ti'),
                                    *connect('ti', 'inverse_shapeof'),
                                    *connect('inverse_shapeof:0', '0:inverse_gather_batch'),
                                    *connect('gather_batch_ind', '1:inverse_gather_batch'),
                                    *connect('gather_axis', '2:inverse_gather_batch'),
                                    *connect('inverse_shapeof:0', '0:inverse_gather_seq', skip_data=True),
                                    *connect('gather_seq_ind', '1:inverse_gather_seq'),
                                    *connect('gather_axis', '2:inverse_gather_seq'),
                                    *connect('inverse_gather_seq', '0:inverse_broadcast'),
                                    *connect('inverse_gather_batch', '1:inverse_broadcast'),
                                    *connect('ti', '0:inverse_reverse', skip_data=True),
                                    *connect('inverse_broadcast', '1:inverse_reverse'),
                                    *connect('inverse_reverse', 'some_op'),
                                    *connect('some_op', 'output')], nodes_with_edges_only=True)

        ReverseTensorIteratorLSTM().find_and_replace_pattern(graph)
        graph.clean_up()

        ref_graph = build_graph(ref_nodes, [*connect('parameter', '0:ti'),
                                            *connect('init_hidden', '1:ti'),
                                            *connect('ti', 'some_op'),
                                            *connect('some_op', 'output')])
        flag, resp = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ti_reverse_const(self):
        graph = build_graph(nodes, [*connect('parameter:0', '0:direct_reverse'),
                                    *connect('seq_len', '1:direct_reverse'),
                                    *connect('direct_reverse', '0:ti'),
                                    *connect('init_hidden', '1:ti'),
                                    *connect('ti', '0:inverse_reverse'),
                                    *connect('inverse_seq_len', '1:inverse_reverse'),
                                    *connect('inverse_reverse', 'some_op'),
                                    *connect('some_op', 'output')], nodes_with_edges_only=True)

        ReverseTensorIteratorLSTM().find_and_replace_pattern(graph)
        graph.clean_up()

        ref_graph = build_graph(ref_nodes, [*connect('parameter', '0:ti'),
                                            *connect('init_hidden', '1:ti'),
                                            *connect('ti', 'some_op'),
                                            *connect('some_op', 'output')])
        flag, resp = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
