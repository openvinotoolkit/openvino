# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from math import sqrt

import numpy as np

from openvino.tools.mo.front.GeLUMerger_Tanh import GeLUMergerTanh
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes_erf = {
    'inp': {'kind': 'op', 'op': 'AnyOp'},
    'mul': {'kind': 'op', 'op': 'Mul'},
    'mul0': {'kind': 'op', 'op': 'Mul'},
    'div': {'kind': 'op', 'op': 'Div'},
    'erf': {'kind': 'op', 'op': 'Erf'},
    'add': {'kind': 'op', 'op': 'Add'},
    'mul_param': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'div_param': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'add_param': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}

nodes_attributes_tanh = {
    'inp': {'kind': 'op', 'op': 'AnyOp'},
    'pow': {'kind': 'op', 'op': 'Pow'},
    'mul': {'kind': 'op', 'op': 'Mul'},
    'mul0': {'kind': 'op', 'op': 'Mul'},
    'mul1': {'kind': 'op', 'op': 'Mul'},
    'mul2': {'kind': 'op', 'op': 'Mul'},
    'tanh': {'kind': 'op', 'op': 'Tanh'},
    'add': {'kind': 'op', 'op': 'Add'},
    'add0': {'kind': 'op', 'op': 'Add'},
    'mul_param': {'kind': 'op',  'type': 'Const', 'op': 'Const'},
    'mul0_param': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'mul1_param': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}

nodes_attributes_ref = {
    'inp': {'kind': 'op', 'op': 'AnyOp'},
    'gelu': {'kind': 'op', 'op': 'Gelu', 'approximation_mode': 'tanh'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}


class TestGeLUMergerReplacement(unittest.TestCase):
    def test_GeLUMergerTanh(self):
        graph = build_graph(nodes_attributes_tanh,
                            [('inp', 'mul2', {'out': 0}),
                             ('inp', 'add',  {'out': 0}),
                             ('inp', 'pow',  {'out': 0}),
                             ('pow', 'mul'),
                             ('mul', 'add'),
                             ('add', 'mul0'),
                             ('mul0', 'tanh'),
                             ('tanh', 'add0'),
                             ('add0', 'mul1'),
                             ('mul1', 'mul2'),
                             ('mul_param', 'mul'),
                             ('mul0_param', 'mul0'),
                             ('mul1_param', 'mul1'),
                             ('mul2', 'out'),
                             ],
                            {'mul0_param': {'shape': np.array([1]), 'value': np.array(sqrt(2.0/3.1415926))},
                             'mul1_param': {'shape': np.array([1]), 'value': np.array(0.5)},
                             'mul_param':  {'shape': np.array([1]), 'value': np.array(0.044715)}
                             },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_attributes_ref,
                            [('inp', 'gelu'),
                             ('gelu', 'out')],
                            {}, nodes_with_edges_only=True)
        graph.stage = 'front'

        replacer = GeLUMergerTanh()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)
