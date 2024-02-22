# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.MakeKaldiConstReshapable import MakeKaldiConstReshapable
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, connect

nodes = {
    **regular_op_with_shaped_data('placeholder_1', [1, 13], {'kind': 'op', 'op': 'Parameter', 'shape': [1, 13]}),
    **regular_op_with_shaped_data('splice_1', [1, 13], {'kind': 'op', 'op': 'Splice',
                                                        'context': np.array([-2, -1, 0, 1, 2])}),
    **regular_op_with_shaped_data('placeholder_2', [1, 26], {'kind': 'op', 'op': None}),
    **regular_op_with_shaped_data('memory_in', [1, 5], {'kind': 'op', 'op': 'ReadValue',
                                                        'shape': int64_array([1, 5])}),
    **regular_op_with_shaped_data('memory_out', [1, 5], {'kind': 'op', 'op': 'Assign', 'shape': int64_array([1, 5])}),
    **result('result'),
    **regular_op_with_shaped_data('crop_in', [1, 4], {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 1, 'dim': 4}),
    **regular_op_with_shaped_data('crop_out', [1, 1], {'kind': 'op', 'op': 'Crop', 'axis': 1, 'offset': 0, 'dim': 1}),
    **regular_op_with_shaped_data('equal', [1, 1], {'kind': 'op', 'op': 'Equal'}),
    **regular_op_with_shaped_data('select', [1, 26], {'kind': 'op', 'op': 'Select'}),
    **regular_op_with_shaped_data('const_0', [1, 1], {'kind': 'op', 'op': 'Const', 'shape': [1, 1],
                                                      'value': [0], 'data_type': np.float32}),
    **regular_op_with_shaped_data('const_1', [1, 1], {'kind': 'op', 'op': 'Const', 'shape': [1, 1],
                                                      'value': [0], 'data_type': np.float32}),
    **regular_op_with_shaped_data('concat', [1, 5], {'kind': 'op', 'op': 'Concat'}),
    **regular_op_with_shaped_data('memory', [1, 26], {'kind': 'op', 'op': 'Assign'}),

    **regular_op_with_shaped_data('shape', None,  {'kind': 'op', 'op': 'ShapeOf'}),
    **regular_op_with_shaped_data('crop_batch', None, {'kind': 'op', 'op': 'Crop', 'offset': int64_array([0])}),
    **regular_op_with_shaped_data('crop_batch_dim', None, {'kind': 'op', 'op': 'Const', 'shape': [1],
                                                             'value': [1], 'data_type': np.int64}),
    **regular_op_with_shaped_data('second_dim', None, {'kind': 'op', 'op': 'Const', 'shape': [1],
                                                      'value': [5], 'data_type': np.int64}),
    **regular_op_with_shaped_data('gather_shape', None, {'kind': 'op', 'op': 'Concat'}),
    **regular_op_with_shaped_data('fill_value', [1, 5], {'kind': 'op', 'op': 'Const', 'shape': [1, 5],
                                                         'value': np.zeros([1, 5]), 'data_type': np.float32}),
    **regular_op_with_shaped_data('fill_value_2', None, {'kind': 'op', 'op': 'Const', 'shape': [1],
                                                         'value': [0], 'data_type': np.float32}),
    **regular_op_with_shaped_data('broadcast', [1, 5], {'kind': 'op', 'op': 'Broadcast'}),

    **regular_op_with_shaped_data('fill_value_ones', [1, 26], {'kind': 'op', 'op': 'Const', 'shape': [1, 26],
                                                               'value': np.zeros([1, 26]), 'data_type': np.int64}),
    **regular_op_with_shaped_data('fill_value_ones_2', [1, 1], {'kind': 'op', 'op': 'Const', 'shape': [1, 1],
                                                                'value': [1], 'data_type': np.int64}),
}


class MakeKaldiConstReshapableTests(unittest.TestCase):

    # graph contains 1 splice with context length 5, should be inserted select with memory as counter with length 5
    def test_reshapable_const(self):
        graph = build_graph(nodes,
                            [*connect('placeholder_1', 'splice_1'),
                             *connect('splice_1', 'placeholder_2'),
                             *connect('placeholder_2', '1:select'),
                             *connect('fill_value', 'memory_in'),
                             *connect('memory_in', 'crop_in'),
                             *connect('crop_in', '0:concat'),
                             *connect('fill_value_ones_2:0', '1:concat'),
                             *connect('concat', 'memory_out'),
                             *connect('memory_out', 'result'),
                             *connect('concat', 'crop_out'),
                             *connect('crop_out', '1:equal'),
                             *connect('fill_value_ones_2:0', '0:equal'),
                             *connect('equal', '0:select'),
                             *connect('fill_value_ones', '2:select'),
                             *connect('select', 'memory')
                             ],
                            nodes_with_edges_only=True)
        graph.strict_mode = False
        MakeKaldiConstReshapable().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes,
                                [*connect('placeholder_1:0', 'splice_1'),
                                 *connect('splice_1', 'placeholder_2'),
                                 *connect('placeholder_2', '1:select'),
                                 *connect('placeholder_1:0', 'shape', skip_data=True),
                                 *connect('shape', '0:crop_batch'),
                                 *connect('crop_batch_dim', '1:crop_batch'),
                                 *connect('second_dim', '1:gather_shape'),
                                 *connect('crop_batch', '0:gather_shape'),
                                 *connect('fill_value_2', '0:broadcast'),
                                 *connect('gather_shape', '1:broadcast'),
                                 *connect('broadcast', 'memory_in'),
                                 *connect('memory_in', 'crop_in'),
                                 *connect('crop_in', '0:concat'),
                                 *connect('fill_value_ones_2', '1:concat'),
                                 *connect('concat', 'memory_out'),
                                 *connect('memory_out', 'result'),
                                 *connect('concat', 'crop_out'),
                                 *connect('crop_out', '1:equal'),
                                 *connect('fill_value_ones_2', '0:equal'),
                                 *connect('equal', '0:select'),
                                 *connect('const_0', '2:select'),
                                 *connect('fill_value_ones', '2:select'),
                                 *connect('select', 'memory')
                                 ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, ref_graph, 'memory')
        self.assertTrue(flag, resp)
