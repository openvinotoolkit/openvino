# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from math import sqrt

from openvino.tools.mo.front.GeLUMerger_Erf import GeLUMergerErf
from openvino.tools.mo.front.common.partial_infer.utils import float_array, int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import const, regular_op, result, build_graph

ref_nodes = {**regular_op('input', {'type': 'Parameter'}),
             **regular_op('gelu', {'type': 'Gelu', 'approximation_mode': 'erf', 'name': 'final_mul'}),
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
        self.assertTrue(graph.get_op_nodes(op='Gelu')[0].approximation_mode == 'erf')
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
        self.assertTrue(graph.get_op_nodes(op='Gelu')[0].approximation_mode == 'erf')
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
        self.assertTrue(graph.get_op_nodes(op='Gelu')[0].approximation_mode == 'erf')
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'Gelu')
