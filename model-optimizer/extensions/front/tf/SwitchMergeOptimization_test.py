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

from extensions.front.tf.SwitchMergeOptimization import SwitchMergeOptimization
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs


class SwitchMergeOptimizationTest(unittest.TestCase):

    def test(self):
        nodes_attributes = {
            'switch_2_input': {'shape': int64_array([1, 3]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'switches_input': {'shape': int64_array([1, 3]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},

            'switch_input_0': {'kind': 'op', 'op': 'SomeOp'},
            'switch_1_input_0': {'kind': 'op', 'op': 'SomeOp'},

            'switch': {'kind': 'op', 'op': 'Switch'},
            'switch_1': {'kind': 'op', 'op': 'Switch'},
            'switch_2': {'kind': 'op', 'op': 'Switch'},

            'some_op': {'kind': 'op', 'op': 'Max'},
            'identity': {'kind': 'op', 'op': 'Identity'},

            'merge': {'kind': 'op', 'op': 'Merge'},

            'select': {'kind': 'op', 'op': 'Select'},

            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
        }

        # check two cases when switch_2 goes to 0-th and 1-st input port of the Merge
        for merge_input_port in range(2):
            graph = build_graph(nodes_attributes,
                                [('switch_2_input', 'switch_2', {'in': 0}),
                                 ('switch_input_0', 'switch', {'in': 0}),
                                 ('switch_1_input_0', 'switch_1', {'in': 0}),
                                 ('switches_input', 'switch', {'in': 1, 'out': 0}),
                                 ('switches_input', 'switch_1', {'in': 1, 'out': 0}),
                                 ('switches_input', 'switch_2', {'in': 1, 'out': 0}),
                                 ('switch', 'some_op', {'in': 0}),
                                 ('switch_1', 'some_op', {'in': 1}),
                                 ('some_op', 'identity', {'in': 0}),
                                 ('switch_2', 'merge', {'in': merge_input_port}),
                                 ('identity', 'merge', {'in': 1 - merge_input_port}),
                                 ('merge', 'last', {'in': 0}),
                                 ], nodes_with_edges_only=True)
            graph.stage = 'front'
            SwitchMergeOptimization().find_and_replace_pattern(graph)

            graph_ref = build_graph(nodes_attributes,
                                    [('switches_input', 'select', {'in': 0}),
                                     ('switch_2_input', 'select', {'in': 1}),
                                     ('switch_input_0', 'some_op', {'in': 0}),
                                     ('switch_1_input_0', 'some_op', {'in': 1}),
                                     ('some_op', 'identity', {'in': 0}),
                                     ('identity', 'select', {'in': 2}),
                                     ('select', 'last', {'in': 0}),
                                     ], nodes_with_edges_only=True)

            (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
            self.assertTrue(flag, resp)
