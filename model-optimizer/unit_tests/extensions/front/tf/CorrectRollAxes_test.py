# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

from extensions.front.tf.CorrectRollAxes import CorrectRollAxes
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'roll': {'kind': 'op', 'op': 'Roll', 'type': 'Roll', 'input_rank_changed': True},
    'roll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'roll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

graph_edges = [
    ('placeholder', 'roll', {'in': 0}),
    ('roll', 'abs'),
    ('abs', 'output'),
    ('roll_shift', 'roll', {'in': 1}),
    ('roll_axes', 'roll', {'in': 2}),
]


ref_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'roll': {'kind': 'op', 'op': 'Roll', 'type': 'Roll'},
    'roll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'roll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    'add': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'mul': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    'less': {'type': 'Less', 'kind': 'op', 'op': 'Less'},
    'zero': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(0)
    },
    'minus_one': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(-1)
    },
}

ref_graph_edges = [
    ('placeholder', 'roll', {'out': 0, 'in': 0}),
    ('roll', 'abs'),
    ('abs', 'output'),
    ('roll_shift', 'roll', {'in': 1}),
    ('mul', 'add', {'in': 1}),
    ('add', 'roll', {'in': 2}),
    ('zero', 'less', {'in': 1}),
    ('minus_one', 'mul', {'in': 1}),
    ('less', 'mul', {'in': 0}),
    ('roll_axes', 'less', {'out': 0, 'in': 0}),
    ('roll_axes', 'add', {'out': 0, 'in': 0}),
]


class CorrectRollAxesTest(unittest.TestCase):
    def test_replacement(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        graph.stage = 'front'
        CorrectRollAxes().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_nonreplacement(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges,
                            update_attributes={'roll': {'input_rank_changed': False}})
        graph.stage = 'front'
        CorrectRollAxes().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges,
                                update_attributes={'roll': {'input_rank_changed': False}})
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
