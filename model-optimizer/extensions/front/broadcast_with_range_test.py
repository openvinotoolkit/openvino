# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.front.broadcast_with_range import ExpandRangeConstant
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect, \
    regular_op_with_empty_data, connect_data


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

        graph_ref = build_graph({
            **regular_op_with_shaped_data('shape', [2], {'type': 'Parameter'}),

            # start
            **valued_const_with_data('start', np.array(0)),
            # limit
            **valued_const_with_data('minus_one', np.array(-1)),
            **valued_const_with_data('zero', np.array(0)),
            **regular_op_with_empty_data('range_dim', {'type': 'Gather'}),
            # delta
            **valued_const_with_data('delta', np.array(1)),
            **regular_op_with_empty_data('range', {'type': 'Range'}),

            # keep dims
            **valued_const_with_data('axes', np.array([0])),
            **regular_op_with_empty_data('keep_shape', {'type': 'Unsqueeze'}),

            **regular_op_with_empty_data('bc', {'type': 'Broadcast'}),
            **result(),
        }, [
            *connect('start', '0:range'),
            *connect('shape', '0:range_dim'),
            *connect('minus_one', '1:range_dim'),
            *connect('zero', '2:range_dim'),
            *connect('range_dim', '1:range'),
            *connect('delta', '2:range'),
            *connect('range', '0:keep_shape'),
            *connect('axes', '1:keep_shape'),
            *connect('keep_shape', '0:bc'),
            *connect_data('shape', '1:bc'),
            *connect('bc', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
