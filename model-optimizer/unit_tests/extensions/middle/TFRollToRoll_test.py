# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.middle.TFRollToRoll import TFRollToRoll
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([8, 40, 56, 2]),
        'kind': 'data',
        'data_type': None
    },
    'shift': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([20, 16]),
        'shape': int64_array([2]),
    },
    'shift_data': {
        'kind': 'data',
        'value': int64_array([20, 16]),
        'shape': int64_array([2]),
    },
    'axes': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([-3, -2]),
        'shape': int64_array([2]),
    },
    'axes_data': {
        'kind': 'data',
        'value': int64_array([-3, -2]),
        'shape': int64_array([2]),
    },
    'tfroll': {'kind': 'op', 'op': 'TFRoll'},
    'tfroll_data': {
        'kind': 'data',
        'shape': int64_array([8, 40, 56, 2]),
        'value': None,
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([8, 40, 56, 2]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('shift', 'shift_data'),
    ('axes', 'axes_data'),
    ('placeholder_data', 'tfroll', {'in': 0}),
    ('shift_data', 'tfroll', {'in': 1}),
    ('axes_data', 'tfroll', {'in': 2}),
    ('tfroll', 'tfroll_data'),
    ('tfroll_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


ref_graph_node_attrs = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([8, 40, 56, 2]),
        'kind': 'data',
        'data_type': None
    },
    'shift': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([20, 16]),
        'shape': int64_array([2]),
    },
    'shift_data': {
        'kind': 'data',
        'value': int64_array([20, 16]),
        'shape': int64_array([2]),
    },
    'axes': {
        'kind': 'op',
        'op': 'Const',
        'type': 'Const',
        'value': int64_array([-3, -2]),
        'shape': int64_array([2]),
    },
    'axes_data': {
        'kind': 'data',
        'value': int64_array([-3, -2]),
        'shape': int64_array([2]),
    },
    'roll': {'kind': 'op', 'op': 'Roll'},
    'roll_data': {
        'kind': 'data',
        'shape': int64_array([8, 40, 56, 2]),
        'value': None,
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {
        'kind': 'data',
        'shape': int64_array([8, 40, 56, 2]),
        'value': None,
    },
    'output': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
}

ref_graph_edges = [
    ('placeholder', 'placeholder_data'),
    ('shift', 'shift_data'),
    ('axes', 'axes_data'),
    ('placeholder_data', 'roll', {'in': 0}),
    ('shift_data', 'roll', {'in': 1}),
    ('axes_data', 'roll', {'in': 2}),
    ('roll', 'roll_data'),
    ('roll_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


class TFRollToRollTest(unittest.TestCase):
    def test_tf_roll_convert(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
        TFRollToRoll().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)