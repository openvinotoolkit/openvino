# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.middle.reverse_tensor_iterator import ReverseTensorIteratorLSTM
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, connect, \
    valued_const_with_data, regular_op_with_empty_data

nodes = {
    **regular_op_with_shaped_data('parameter', [1, 3, 227, 227],
                                  {'type': 'Parameter', 'op': 'Parameter', 'shape': [1, 3, 227, 227]}),
    **valued_const_with_data('direct_reverse_seqlen', np.array([227])),
    **regular_op_with_shaped_data('direct_reverse', [1, 3, 227, 227], {'type': 'ReverseSequence', 'op': 'ReverseSequence',
                                                    'seq_axis': 2, 'batch_axis': 0}),
    **regular_op_with_empty_data('init_hidden', {'type': 'Init', 'op': 'Init'}),
    **regular_op_with_shaped_data('ti', [1, 2, 34, 56], {'type': 'TensorIterator', 'op': 'TensorIterator',
                                        'output_port_map': [{'axis': 2, 'start': 0, 'end': -1, 'stride': 1,
                                                             'external_port_id': 0}],
                                        'input_port_map': [{'axis': 2, 'start': -1, 'end': 0, 'stride': -1,
                                                            'external_port_id': 0}]}),
    **valued_const_with_data('inverse_reverse_seqlen', np.array([34])),
    **regular_op_with_shaped_data('inverse_reverse', [1, 2, 34, 56], {'type': 'ReverseSequence',
                                                                      'op': 'ReverseSequence',
                                                                      'seq_axis': 2, 'batch_axis': 0}),
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
