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
from generator import generator, generate

from extensions.front.kaldi.tdnn_component_replacer import TdnnComponentReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op, result, connect_front, const


@generator
class TdnnComponentReplacerTest(unittest.TestCase):

    @generate(*[
        ([[1, 1, 1], [4, 4, 4]], [1, 2], [-1, 1],),
        ([[1, 1, 1], [4, 4, 4]], [1, 2], [-1, 1, 2, 10, 1000],),
        ([[1, 1, 1], [4, 4, 4]], [1, 2], [-1, 0]),
    ])
    def test_tdnnreplacer(self, weights, biases, time_offsets):
        def generate_offsets():
            offset_edges = []
            offset_nodes = {}

            for i, t in enumerate(time_offsets):
                offset_nodes.update(**regular_op('memoryoffset_' + str(i), {'type': None}))

                if t != 0:
                    offset_edges.append(('placeholder', 'memoryoffset_' + str(i), {'out': 0, 'in': 0}))
                    offset_edges.append(('memoryoffset_' + str(i), 'concat', {'out': 0, 'in': i}))
                else:
                    offset_edges.append(('placeholder', 'concat', {'out': 0, 'in': i}))

            return offset_nodes, offset_edges

        offset_nodes, ref_offset_edges = generate_offsets()

        nodes = {
            **offset_nodes,
            **regular_op('placeholder', {'type': 'Parameter'}),
            **regular_op('tdnncomponent', {'op': 'tdnncomponent',
                                           'weights': np.array(weights),
                                           'biases': np.array(biases),
                                           'time_offsets': np.array(time_offsets)}),
            **const('weights', np.array(weights)),
            **const('biases', np.array(biases)),
            **regular_op('concat', {'type': 'Concat', 'axis': 1}),
            **regular_op('memoryoffset_0', {'type': None}),
            **regular_op('memoryoffset_1', {'type': None}),
            **regular_op('memoryoffset_2', {'type': None}),
            **regular_op('fully_connected', {'type': 'FullyConnected'}),
            **result('result'),
        }

        graph = build_graph(nodes, [
            *connect_front('placeholder', 'tdnncomponent'),
            *connect_front('tdnncomponent', 'result')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        ref_graph = build_graph(nodes, [
            *ref_offset_edges,
            *connect_front('concat', '0:fully_connected'),
            *connect_front('weights', '1:fully_connected'),
            *connect_front('biases', '2:fully_connected'),
            *connect_front('fully_connected', 'result')
        ], nodes_with_edges_only=True)

        TdnnComponentReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
