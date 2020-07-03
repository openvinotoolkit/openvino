"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.front.LayerNorm import LayerNorm
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes_mvn = {
    'inp': {'kind': 'op', 'op': 'AnyOp'},
    'pool0': {'kind': 'op', 'op': 'ReduceMean'},
    'pool1': {'kind': 'op', 'op': 'ReduceMean'},
    'pow': {'kind': 'op', 'op': 'Pow'},
    'div': {'kind': 'op', 'op': 'Div'},
    'sqrt': {'kind': 'op', 'op': 'Pow'},
    'add': {'kind': 'op', 'op': 'Add'},
    'sub': {'kind': 'op', 'op': 'Sub'},
    'add_param': {'kind': 'op', 'op': 'Const'},
    'pow_param': {'kind': 'op', 'op': 'Const'},
    'pool0_param': {'kind': 'op', 'op': 'Const'},
    'pool1_param': {'kind': 'op', 'op': 'Const'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}

nodes_attributes_ref = {
    'inp': {'kind': 'op', 'op': 'AnyOp'},
    'mvn': {'kind': 'op', 'op': 'MVN'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}


class TestMVNPatternReplacement(unittest.TestCase):
    def test_MVNPatternReplacement_test_1(self):
        graph = build_graph(nodes_attributes_mvn,
                            [('inp', 'pool0', {'out': 0}),
                             ('inp', 'sub', {'out': 0}),
                             ('pool0', 'sub'),
                             ('sub', 'pow'),
                             ('pow', 'pool1'),
                             ('pool1', 'add'),
                             ('add', 'sqrt'),
                             ('sqrt', 'div'),
                             ('sub', 'div'),
                             ('div', 'out'),
                             ('pow_param', 'sqrt'),
                             ('add_param', 'add'),
                             ('pool0_param', 'pool0'),
                             ('pool1_param', 'pool1'),
                             ],
                            {'pow_param': {'shape': np.array([1]), 'value': np.array(0.5)},
                             'add_param': {'shape': np.array([1]), 'value': np.array(1e-06)},
                             'pool0_param': {'shape': np.array([1]), 'value': np.array(-1)},
                             'pool1_param': {'shape': np.array([1]), 'value': np.array(-1)},
                             },
                            nodes_with_edges_only=True)
        graph_ref = build_graph(nodes_attributes_ref,
                                [('inp', 'mvn'),
                                 ('mvn', 'out')],
                                {}, nodes_with_edges_only=True)
        graph.stage = 'front'

        replacer = LayerNorm()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)
