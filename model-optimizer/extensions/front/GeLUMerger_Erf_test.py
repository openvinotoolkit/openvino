"""
 Copyright (C) 2018-2021 Intel Corporation

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
from math import sqrt

from extensions.front.GeLUMerger_Erf import GeLUMergerErf
from mo.front.common.partial_infer.utils import float_array, int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const, regular_op, result, build_graph

ref_nodes = {**regular_op('input', {'type': 'Parameter'}),
             **regular_op('gelu', {'type': 'Gelu', 'name': 'final_mul'}),
             **result('result')
             }
ref_edges = [('input', 'gelu'), ('gelu', 'result')]


class GeLUMergerErfTest(unittest.TestCase):
    nodes = {
        **regular_op('input', {'op': 'Parameter', 'type': 'Parameter'}),
        **regular_op('mul', {'op': 'Mul'}),
        **regular_op('mul0', {'op': 'Mul', 'name': 'final_mul'}),
        **regular_op('div', {'op': 'Div'}),
        **regular_op('erf', {'op': 'Erf'}),
        **regular_op('add', {'op': 'Add'}),
        **const('mul_param', float_array([0.5])),
        **const('div_param', float_array([sqrt(2.)])),
        **const('add_param', int64_array([1])),
        **result('result'),
    }

    def test_gelu_p1(self):
        edges = [('input', 'mul'),
                 ('mul', 'mul0'),
                 ('input', 'div'),
                 ('div', 'erf'),
                 ('erf', 'add'),
                 ('add', 'mul0'),
                 ('mul_param', 'mul'),
                 ('div_param', 'div'),
                 ('add_param', 'add'),
                 ('mul0', 'result')]

        graph = build_graph(self.nodes, edges)

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        GeLUMergerErf().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'Gelu')

    def test_gelu_p2(self):
        edges = [('input', 'mul'),
                 ('div', 'erf'),
                 ('erf', 'add'),
                 ('add', 'mul'),
                 ('mul', 'mul0'),
                 ('mul_param', 'mul0'),
                 ('div_param', 'div'),
                 ('add_param', 'add'),
                 ('mul0', 'result')]

        graph = build_graph(self.nodes, edges)

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        GeLUMergerErf().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'Gelu')

    def test_gelu_p3(self):
        edges = [('input', 'mul'),
                 ('div', 'erf'),
                 ('erf', 'add'),
                 ('add', 'mul'),
                 ('mul', 'mul0'),
                 ('mul_param', 'mul'),
                 ('div_param', 'div'),
                 ('add_param', 'add'),
                 ('mul0', 'result')]

        graph = build_graph(self.nodes, edges)

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        GeLUMergerErf().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'Gelu')
