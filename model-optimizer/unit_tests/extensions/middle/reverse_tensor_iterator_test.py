# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.middle.reverse_tensor_iterator import ReverseTensorIteratorLSTM, ReverseTensorIteratorLSTMWithSqueeze
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, connect, \
    valued_const_with_data, regular_op_with_empty_data

nodes = {
    **regular_op_with_shaped_data('parameter', [1, 3, 227, 227],
                                  {'type': 'Parameter', 'op': 'Parameter', 'shape': [1, 3, 227, 227]}),
    **valued_const_with_data('direct_reverse_seqlen', np.array([227])),
    **regular_op_with_shaped_data('direct_unsqueeze', [1, 1, 3, 227, 227], {'type': 'Unsqueeze', 'op': 'Unsqueeze', 'axis': 0}),
    **regular_op_with_shaped_data('direct_reverse', [1, 3, 227, 227], {'type': 'ReverseSequence', 'op': 'ReverseSequence',
                                                    'seq_axis': 2, 'batch_axis': 0}),
    **regular_op_with_shaped_data('direct_squeeze', [1, 3, 227, 227], {'type': 'Squeeze', 'op': 'Squeeze', 'axis': 0}),
    **regular_op_with_empty_data('init_hidden', {'type': 'Init', 'op': 'Init'}),
    **regular_op_with_shaped_data('ti', [1, 2, 34, 56], {'type': 'TensorIterator', 'op': 'TensorIterator',
                                        'output_port_map': [{'axis': 2, 'start': 0, 'end': -1, 'stride': 1,
                                                             'external_port_id': 0}],
                                        'input_port_map': [{'axis': 2, 'start': -1, 'end': 0, 'stride': -1,
                                                            'external_port_id': 0}]}),
    **valued_const_with_data('inverse_reverse_seqlen', np.array([34])),
    **regular_op_with_shaped_data('inverse_unsqueeze', [1, 1, 2, 34, 56], {'type': 'Unsqueeze', 'op': 'Unsqueeze', 'axis': 0}),
    **regular_op_with_shaped_data('inverse_reverse', [1, 2, 34, 56], {'type': 'ReverseSequence',
                                                                      'op': 'ReverseSequence',
                                                                      'seq_axis': 2, 'batch_axis': 0}),
    **regular_op_with_shaped_data('inverse_squeeze', [1, 2, 34, 56], {'type': 'Squeeze', 'op': 'Squeeze', 'axis': 0}),
    **regular_op_with_empty_data('some_op', {'op': 'SomeOp'})
}

ref_nodes = {
    **regular_op_with_shaped_data('parameter', [1, 3, 227, 227],
                                  {'type': 'Parameter', 'op': 'Parameter', 'shape': [1, 3, 227, 227]}),
    **regular_op_with_empty_data('init_hidden', {'type': 'Init', 'op': 'Init'}),
    **regular_op_with_empty_data('ti', {'type': 'TensorIterator', 'op': 'TensorIterator',
                                        'output_port_map': [{'axis': 2, 'start': -1, 'end': 0, 'stride': -1,
                                                             'external_port_id': 0}],
                                        'input_port_map': [{'axis': 2, 'start': None, 'end': None, 'stride': 1,
                                                            'external_port_id': 0}]}),
    **regular_op_with_empty_data('some_op', {'op': 'SomeOp'})
}


class ReverseTensorIteratorTest(unittest.TestCase):
    def test_ti_reverse(self):
        graph = build_graph(nodes, [*connect('parameter', '0:direct_reverse'),
                                    *connect('direct_reverse_seqlen', '1:direct_reverse'),
                                    *connect('direct_reverse', '0:ti'),
                                    *connect('init_hidden', '1:ti'),
                                    *connect('ti', '0:inverse_reverse'),
                                    *connect('inverse_reverse_seqlen', '1:inverse_reverse'),
                                    *connect('inverse_reverse', 'some_op')], nodes_with_edges_only=True)
        ReverseTensorIteratorLSTM().find_and_replace_pattern(graph)

        ref_graph = build_graph(ref_nodes, [*connect('parameter', '0:ti'),
                                            *connect('init_hidden', '1:ti'),
                                            *connect('ti', 'some_op')])
        flag, resp = compare_graphs(graph, ref_graph, 'some_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_ti_reverse_squeeze(self):
        graph = build_graph(nodes, [*connect('parameter', '0:direct_unsqueeze'),
                                    *connect('direct_unsqueeze', '0:direct_reverse'),
                                    *connect('direct_reverse_seqlen', '1:direct_reverse'),
                                    *connect('direct_reverse', 'direct_squeeze'),
                                    *connect('direct_squeeze', '0:ti'),
                                    *connect('init_hidden', '1:ti'),
                                    *connect('ti', 'inverse_unsqueeze'),
                                    *connect('inverse_unsqueeze', '0:inverse_reverse'),
                                    *connect('inverse_reverse_seqlen', '1:inverse_reverse'),
                                    *connect('inverse_reverse', 'inverse_squeeze'),
                                    *connect('inverse_squeeze', 'some_op')], nodes_with_edges_only=True)
        direct_reverse = Node(graph, 'direct_reverse')
        direct_reverse.out_port(0).data.set_shape([1, 1, 3, 227, 227])
        direct_reverse.seq_axis = 3
        inverse_reverse = Node(graph, 'inverse_reverse')
        inverse_reverse.out_port(0).data.set_shape([1, 1, 2, 34, 56])
        inverse_reverse.seq_axis = 3
        ReverseTensorIteratorLSTMWithSqueeze().find_and_replace_pattern(graph)
        ref_graph = build_graph(ref_nodes, [*connect('parameter', 'ti'),
                                            *connect('init_hidden', '1:ti'),
                                            *connect('ti', 'some_op')])
        flag, resp = compare_graphs(graph, ref_graph, 'some_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
