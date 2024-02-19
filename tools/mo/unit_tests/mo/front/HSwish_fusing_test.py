# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.HSwish_fusion import HSwishWithClamp, HSwishWithMinMax
from openvino.tools.mo.front.common.partial_infer.utils import float_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const, regular_op, result, build_graph_with_edge_attrs

ref_nodes = {**regular_op('input', {'type': 'Parameter'}),
             **regular_op('hswish', {'type': 'HSwish', 'name': 'final_mul'}),
             **result('result')
             }
ref_edges = [('input', 'hswish'), ('hswish', 'result')]


class HSwishWithClampTest(unittest.TestCase):
    nodes = {
        **regular_op('input', {'type': 'Parameter'}),
        **regular_op('add', {'op': 'Add'}),
        **regular_op('relu6', {'op': 'Clamp'}),
        **regular_op('mul', {'op': 'Mul'}),
        **regular_op('mul_2', {'op': 'Mul', 'name': 'final_mul'}),
        **const('const_0', float_array([0.0])),
        **const('const_3', float_array([3.0])),
        **const('const_6', float_array([6.0])),
        **const('const_1_6', float_array([1.0 / 6.0])),
        **result('result'),
    }

    edges = [('input', 'mul', {'in': 0, 'out': 0}),
             ('input', 'add', {'in': 0, 'out': 0}),
             ('const_3', 'add', {'in': 1, 'out': 0}),
             ('add', 'relu6', {'in': 0, 'out': 0}),
             ('const_0', 'relu6', {'in': 1, 'out': 0}),
             ('const_6', 'relu6', {'in': 2, 'out': 0}),
             ('relu6', 'mul', {'in': 1, 'out': 0}),
             ('mul', 'mul_2', {'in': 0, 'out': 0}),
             ('const_1_6', 'mul_2', {'in': 1, 'out': 0}),
             ('mul_2', 'result', {'in': 0, 'out': 0})]

    def test_hswish_with_clamp(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {})

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        HSwishWithClamp().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'HSwish')

    def test_hswish_with_clamp_wrong_constant(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {'const_0': {'value': float_array([0.00001])}})

        graph_ref = graph.copy()
        graph.stage = 'front'

        HSwishWithClamp().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_hswish_with_clamp_different_tensors(self):
        graph = build_graph_with_edge_attrs({
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('input_2', {'type': 'Parameter'}),
            **regular_op('add', {'op': 'Add'}),
            **regular_op('relu6', {'op': 'Clamp'}),
            **regular_op('mul', {'op': 'Mul'}),
            **regular_op('mul_2', {'op': 'Mul', 'name': 'final_mul'}),
            **const('const_0', float_array([0.0])),
            **const('const_3', float_array([3.0])),
            **const('const_6', float_array([6.0])),
            **const('const_1_6', float_array([1.0 / 6.0])),
            **result('result'),
        }, [('input', 'mul', {'in': 0, 'out': 0}),
            ('input_2', 'add', {'in': 0, 'out': 0}),
            ('const_3', 'add', {'in': 1, 'out': 0}),
            ('add', 'relu6', {'in': 0, 'out': 0}),
            ('const_0', 'relu6', {'in': 1, 'out': 0}),
            ('const_6', 'relu6', {'in': 2, 'out': 0}),
            ('relu6', 'mul', {'in': 1, 'out': 0}),
            ('mul', 'mul_2', {'in': 0, 'out': 0}),
            ('const_1_6', 'mul_2', {'in': 1, 'out': 0}),
            ('mul_2', 'result', {'in': 0, 'out': 0})])

        graph_ref = graph.copy()
        graph.stage = 'front'

        HSwishWithClamp().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)


class HSwishWithMinMaxTest(unittest.TestCase):
    nodes = {
        **regular_op('input', {'type': 'Parameter'}),
        **regular_op('add', {'op': 'Add'}),
        **regular_op('max', {'op': 'Maximum'}),
        **regular_op('min', {'op': 'Minimum'}),
        **regular_op('mul', {'op': 'Mul'}),
        **regular_op('mul_2', {'op': 'Mul', 'name': 'final_mul'}),
        **const('const_0', float_array([0.0])),
        **const('const_3', float_array([3.0])),
        **const('const_6', float_array([6.0])),
        **const('const_1_6', float_array([1.0 / 6.0])),
        **result('result'),
    }

    edges = [('input', 'mul', {'in': 1, 'out': 0}),
             ('input', 'add', {'in': 0, 'out': 0}),
             ('const_3', 'add', {'in': 1, 'out': 0}),
             ('add', 'max', {'in': 0, 'out': 0}),
             ('const_0', 'max', {'in': 1, 'out': 0}),
             ('max', 'min', {'in': 0, 'out': 0}),
             ('const_6', 'min', {'in': 1, 'out': 0}),
             ('min', 'mul', {'in': 0, 'out': 0}),
             ('mul', 'mul_2', {'in': 0, 'out': 0}),
             ('const_1_6', 'mul_2', {'in': 1, 'out': 0}),
             ('mul_2', 'result', {'in': 0, 'out': 0})]

    def test_hswish_with_min_max(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {})

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        HSwishWithMinMax().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'HSwish')

    def test_hswish_with_min_max_wrong_constant(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {'const_0': {'value': float_array([0.00001])}})

        graph_ref = graph.copy()
        graph.stage = 'front'

        HSwishWithMinMax().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_hswish_with_min_max_different_tensors(self):
        graph = build_graph_with_edge_attrs({
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('input_2', {'type': 'Parameter'}),
            **regular_op('add', {'op': 'Add'}),
            **regular_op('max', {'op': 'Maximum'}),
            **regular_op('min', {'op': 'Minimum'}),
            **regular_op('mul', {'op': 'Mul'}),
            **regular_op('mul_2', {'op': 'Mul', 'name': 'final_mul'}),
            **const('const_0', float_array([0.0])),
            **const('const_3', float_array([3.0])),
            **const('const_6', float_array([6.0])),
            **const('const_1_6', float_array([1.0 / 6.0])),
            **result('result'),
        }, [('input_2', 'mul', {'in': 1, 'out': 0}),
            ('input', 'add', {'in': 0, 'out': 0}),
            ('const_3', 'add', {'in': 1, 'out': 0}),
            ('add', 'max', {'in': 0, 'out': 0}),
            ('const_0', 'max', {'in': 1, 'out': 0}),
            ('max', 'min', {'in': 0, 'out': 0}),
            ('const_6', 'min', {'in': 1, 'out': 0}),
            ('min', 'mul', {'in': 0, 'out': 0}),
            ('mul', 'mul_2', {'in': 0, 'out': 0}),
            ('const_1_6', 'mul_2', {'in': 1, 'out': 0}),
            ('mul_2', 'result', {'in': 0, 'out': 0})])

        graph_ref = graph.copy()
        graph.stage = 'front'

        HSwishWithMinMax().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
