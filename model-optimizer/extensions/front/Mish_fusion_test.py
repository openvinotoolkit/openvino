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

from extensions.front.Mish_fusion import MishFusion
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op, result, build_graph_with_edge_attrs

ref_nodes = {**regular_op('input', {'type': 'Parameter'}),
             **regular_op('mish', {'type': 'Mish', 'name': 'final_mul'}),
             **result('result')
             }
ref_edges = [('input', 'mish'), ('mish', 'result')]


class MishFusionTest(unittest.TestCase):
    nodes = {
        **regular_op('input', {'type': 'Parameter'}),
        **regular_op('softplus', {'op': 'SoftPlus'}),
        **regular_op('tanh', {'op': 'Tanh'}),
        **regular_op('mul', {'op': 'Mul', 'name': 'final_mul'}),
        **result('result'),
    }

    edges = [('input', 'softplus', {'in': 0, 'out': 0}),
             ('input', 'mul', {'in': 0, 'out': 0}),
             ('softplus', 'tanh', {'in': 0, 'out': 0}),
             ('tanh', 'mul', {'in': 1, 'out': 0}),
             ('mul', 'result', {'in': 0, 'out': 0})]

    def test_mish_fusion(self):
        graph = build_graph_with_edge_attrs(self.nodes, self.edges, {})

        graph_ref = build_graph(ref_nodes, ref_edges)
        graph.stage = 'front'

        MishFusion().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='final_mul')) == 1 and
                        graph.get_op_nodes(name='final_mul')[0].op == 'Mish')

    def test_mish_fusion_different_source(self):
        # check case when different tensors goes to Mul and SoftPlus
        graph = build_graph_with_edge_attrs({
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('input_2', {'type': 'Parameter'}),
            **regular_op('softplus', {'op': 'SoftPlus'}),
            **regular_op('tanh', {'op': 'Tanh'}),
            **regular_op('mul', {'op': 'Mul', 'name': 'final_mul'}),
            **result('result'),
        }, [('input', 'softplus', {'in': 0, 'out': 0}),
            ('input_2', 'mul', {'in': 0, 'out': 0}),
            ('softplus', 'tanh', {'in': 0, 'out': 0}),
            ('tanh', 'mul', {'in': 1, 'out': 0}),
            ('mul', 'result', {'in': 0, 'out': 0})], {})

        graph_ref = graph.copy()
        graph.stage = 'front'

        MishFusion().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
