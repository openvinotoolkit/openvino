# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.broadcast_with_range import ExpandRangeConstant
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect, \
    regular_op_with_empty_data


class TestRangeBroadcast(unittest.TestCase):
    def test_broadcast_with_range_positive_test(self):
        graph = build_graph({
            **regular_op_with_shaped_data('shape', [2], {'type': 'Parameter'}),
            **valued_const_with_data('value', np.arange(0, 384).reshape((1, 384))),
            **regular_op_with_empty_data('bc', {'type': 'Broadcast'}),
            **result(),
        }, [
            *connect('value', '0:bc'),
            *connect('shape', '1:bc'),
            *connect('bc', 'output'),
        ], nodes_with_edges_only=True)
        ExpandRangeConstant().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs={
            **regular_op_with_shaped_data('shape', [2], {'type': 'Parameter'}),
            **valued_const_with_data('value', np.arange(0, 384).reshape((1, 384))),
            **regular_op_with_empty_data('bc', {'type': 'Broadcast'}),
            **regular_op_with_empty_data('shapeof', {'type': 'ShapeOf'}),
            **regular_op_with_empty_data('select', {'type': 'Select'}),
            **regular_op_with_empty_data('gather', {'type': 'Gather'}),
            'gather_const': {'type': 'Gather', 'kind': 'op', 'op': 'Gather'},
            'equal': {'type': 'Equal', 'kind': 'op', 'op': 'Equal'},

            # start
            **valued_const_with_data('start', np.array(0)),
            # limit
            **valued_const_with_data('minus_one_0', np.array(-1)),
            **valued_const_with_data('zero_0', np.array(0)),
            **valued_const_with_data('minus_one_1', np.array(-1)),
            **valued_const_with_data('zero_1', np.array(0)),
            # delta
            **valued_const_with_data('delta', np.array(1)),
            **regular_op_with_shaped_data('range', [1, 384], {'type': 'Range'}),

            # keep dims
            **valued_const_with_data('axes', np.array([0])),
            **regular_op_with_shaped_data('keep_shape', [1, 384], {'type': 'Unsqueeze'}),

            **valued_const_with_data('one', np.array(1)),

            **result(),
        },
            edges=[
                *connect('value', 'shapeof'),
                *connect('gather', '0:equal'),
                ('gather', 'select', {'in': 2, 'out': 0}),
                ('gather_const', 'select', {'in': 1}),
                ('equal', 'select', {'in': 0}),
                *connect('minus_one_0', '1:gather'),
                *connect('zero_0', '2:gather'),
                *connect('shapeof', '0:gather_const'),
                *connect('minus_one_1', '1:gather_const'),
                *connect('zero_1', '2:gather_const'),
                *connect('start', '0:range'),
                *connect('select', '1:range'),
                *connect('delta', '2:range'),
                *connect('range', '0:keep_shape'),
                *connect('axes', '1:keep_shape'),
                *connect('keep_shape', '0:bc'),
                *connect('one', '1:equal'),
                *connect('shape', '1:bc'),
                ('shape_d', 'gather', {'out': 0, 'in': 0}),
                *connect('bc', 'output'),
            ],
            update_attributes={
                'range_d': {'value': np.arange(0, 384).reshape((1, 384))},
                'keep_shape_d': {'value': np.arange(0, 384).reshape((1, 384))},
            })

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
