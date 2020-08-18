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
from extensions.front.kaldi.tdnn_component_replacer import TdnnComponentReplacer
import numpy as np
from generator import generator, generate

from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op, result, connect_front


@generator
class TdnnComponentReplacerTest(unittest.TestCase):

    @generate(*[
        ([[1, 1, 1], [4, 4, 4]], [1, 2], [-1, 0]),
    ])
    def test_tdnnreplacer(self, weights, biases, time_offsets):
        nodes = {
            **regular_op('placeholder', {'type': 'Parameter'}),
            **regular_op('tdnncomponent', {'op': 'tdnncomponent',
                                           'weights': np.array(weights),
                                           'biases': np.array(biases),
                                           'time_offsets': np.array(time_offsets)}),
            **regular_op('concat', {'axis': 0}),
            **regular_op('memoryoffset_1', {}),
            **regular_op('memoryoffset_2', {}),
            **regular_op('conv', {}),
            **result('result'),
        }

        graph = build_graph(nodes, [
            *connect_front('placeholder', 'tdnncomponent'),
            *connect_front('tdnncomponent', 'result')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        ref_graph = build_graph(nodes, [
            *connect_front('placeholder', 'memoryoffset_1'),
            *connect_front('placeholder', 'memoryoffset_2'),
            *connect_front('memoryoffset_1', '0:concat'),
            *connect_front('memoryoffset_2', '1:concat'),
            *connect_front('concat', 'conv'),
            *connect_front('conv', 'result')
        ], nodes_with_edges_only=True)

        TdnnComponentReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=False)
        self.assertTrue(flag, resp)

