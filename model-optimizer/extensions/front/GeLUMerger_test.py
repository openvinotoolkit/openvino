"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

import numpy as np
from math import sqrt

from extensions.front.GeLUMerger_Erf import GeLUMergerErf
from extensions.front.GeLUMerger_Tanh import GeLUMergerTanh
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

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
    'gelu': {'kind': 'op', 'op': 'Gelu'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}

class TestGeLUMergerReplacement(unittest.TestCase):
    def test_GeLUMergerErf_test_1(self):
        graph = build_graph(nodes_attributes_erf,
                            [('inp', 'mul0', {'out': 0}),
                             ('inp', 'div',  {'out': 0}),
                             ('mul', 'mul0'),
                             ('div', 'erf'),
                             ('erf', 'add'),
                             ('add', 'mul0'),
                             ('mul_param', 'mul'),
                             ('div_param', 'div'),
                             ('add_param', 'add'),
                             ('mul0', 'out'),
                             ],
                            {'mul_param': {'shape': np.array([1]), 'value': np.array(0.5)},
                             'add_param': {'shape': np.array([1]), 'value': np.array(1.0)},
                             'div_param': {'shape': np.array([1]), 'value': np.array(sqrt(2.0))}
                             },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_attributes_ref,
                            [('inp', 'gelu'),
                             ('gelu', 'out')],
                            {}, nodes_with_edges_only=True)
        graph.stage = 'front'

        replacer = GeLUMergerErf()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_GeLUMergerTanh_test_2(self):
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