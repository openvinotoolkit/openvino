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

from extensions.front.ReduceL2Decomposition import ReduceL2Decomposition
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const

nodes_attributes = {
    'input': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'reduce_l2': {'type': None, 'kind': 'op', 'op': 'ReduceL2', 'axis': 0, 'name': 'my_reduce', 'keep_dims': 0},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    # new layers
    'mul': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    'reduce_sum': {'type': 'ReduceSum', 'kind': 'op', 'op': 'ReduceSum', 'axis': 0, 'keep_dims': 0},
    'pow': {'type': 'Power', 'kind': 'op', 'op': 'Pow'},
    **const('half', np.array(0.5, dtype=np.float32)),
}


class ReduceL2DecompositionTest(unittest.TestCase):
    def test(self):
        graph = build_graph(nodes_attributes,
                            [('input', 'reduce_l2', {'in': 0, 'out': 0}),
                             ('reduce_l2', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('input', 'mul', {'in': 0, 'out': 0}),
                                 ('input', 'mul', {'in': 1, 'out': 0}),
                                 ('mul', 'reduce_sum', {'in': 0, 'out': 0}),
                                 ('reduce_sum', 'pow', {'in': 0, 'out': 0}),
                                 ('half', 'pow', {'in': 1, 'out': 0}),
                                 ('pow', 'result', {'in': 0, 'out': 0}),
                                 ],
                                {}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        ReduceL2Decomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Pow')[0]]['name'] == 'my_reduce')
