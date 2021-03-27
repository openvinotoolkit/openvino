# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.front.Swish_fusion import SwishWithSigmoidWithoutBeta, SwishWithSigmoidWithBeta
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op, result, build_graph_with_edge_attrs

ref_nodes = {**regular_op('input', {'type': 'Parameter'}),
             **regular_op('swish', {'type': 'Swish', 'name': 'final_mul'}),
             **result('result')
             }
ref_edges = [('input', 'swish'), ('swish', 'result')]


class SwishWithSigmoidWithoutBetaTest(unittest.TestCase):
    nodes = {
        **regular_op('input', {'type': 'Parameter'}),
        **regular_op('sigmoid', {'op': 'Sigmoid'}),
        **regular_op('mul', {'op': 'Mul', 'name': 'final_mul'}),
        **result('result'),
    }

    edges = [('input', 'mul', {'in': 0, 'out': 0}),
             ('input', 'sigmoid', {'in': 0, 'out': 0}),
             ('sigmoid', 'mul', {'in': 1, 'out': 0}),
             ('mul', 'result', {'in': 0, 'out': 0})]

    def test_swish_with_sigmoid_without_beta_test(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {})

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        SwishWithSigmoidWithoutBeta().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'Swish')

    def test_swish_with_sigmoid_without_beta_different_tensors(self):
        graph = build_graph_with_edge_attrs({
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('input_2', {'type': 'Parameter'}),
            **regular_op('sigmoid', {'op': 'Sigmoid'}),
            **regular_op('mul', {'op': 'Mul', 'name': 'final_mul'}),
            **result('result'),
        }, [('input_2', 'mul', {'in': 0, 'out': 0}),
            ('input', 'sigmoid', {'in': 0, 'out': 0}),
            ('sigmoid', 'mul', {'in': 1, 'out': 0}),
            ('mul', 'result', {'in': 0, 'out': 0})], {})

        graph_ref = graph.copy()
        graph.stage = 'front'

        SwishWithSigmoidWithoutBeta().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)


class SwishWithSigmoidWithBetaTest(unittest.TestCase):
    nodes = {
        **regular_op('input', {'type': 'Parameter'}),
        **regular_op('beta', {'type': 'Parameter'}),
        **regular_op('mul_beta', {'op': 'Mul'}),
        **regular_op('sigmoid', {'op': 'Sigmoid'}),
        **regular_op('mul_2', {'op': 'Mul', 'name': 'final_mul'}),
        **result('result'),
    }

    edges = [('input', 'mul_beta', {'in': 0, 'out': 0}),
             ('input', 'mul_2', {'in': 0, 'out': 0}),
             ('beta', 'mul_beta', {'in': 1, 'out': 0}),
             ('mul_beta', 'sigmoid', {'in': 0, 'out': 0}),
             ('sigmoid', 'mul_2', {'in': 1, 'out': 0}),
             ('mul_2', 'result', {'in': 0, 'out': 0})]

    def test_swish_with_sigmoid_with_beta_test(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {})

        new_ref_nodes = ref_nodes.copy()
        new_ref_nodes.update(**regular_op('beta', {'type': 'Parameter'}))

        graph_ref = build_graph(new_ref_nodes, ref_edges + [('beta', 'swish')])
        graph.stage = 'front'

        SwishWithSigmoidWithBeta().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'Swish')

    def test_swish_with_sigmoid_with_beta_different_tensors(self):
        graph = build_graph_with_edge_attrs({
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('input_2', {'type': 'Parameter'}),
            **regular_op('beta', {'type': 'Parameter'}),
            **regular_op('mul_beta', {'op': 'Mul'}),
            **regular_op('sigmoid', {'op': 'Sigmoid'}),
            **regular_op('mul_2', {'op': 'Mul', 'name': 'final_mul'}),
            **result('result'),
        }, [('input', 'mul_beta', {'in': 0, 'out': 0}),
            ('input_2', 'mul_2', {'in': 0, 'out': 0}),
            ('beta', 'mul_beta', {'in': 1, 'out': 0}),
            ('mul_beta', 'sigmoid', {'in': 0, 'out': 0}),
            ('sigmoid', 'mul_2', {'in': 1, 'out': 0}),
            ('mul_2', 'result', {'in': 0, 'out': 0})], {})

        graph_ref = graph.copy()
        graph.stage = 'front'

        SwishWithSigmoidWithBeta().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
