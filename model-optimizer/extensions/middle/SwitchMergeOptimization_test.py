"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.middle.SwitchMergeOptimization import SwitchMergeMiddleReplacer
from mo.utils.unittest.graph import build_graph_with_attrs, compare_graphs


class SwitchMergeOptimizationTest(unittest.TestCase):
    # Test for case when we tile 2 dimensions
    def test(self):
        pattern_matcher = SwitchMergeMiddleReplacer()
        pattern = pattern_matcher.pattern()

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'],
                                       edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[('Merge_output', {'kind': 'data',}),
                                                             ('Merge_output_op', {'kind': 'op', }),
                                                             ('Switch_1_input', dict(kind='data')),
                                                             ('Switch_input', dict(kind='data')),
                                                             ('Switch_1_input_op', dict(kind='op')),
                                                             ('Switch_input_op', dict(kind='op')),
                                                             ],
                                       new_edges_with_attrs=[('Merge', 'Merge_output'),
                                                             ('Merge_output', 'Merge_output_op'),
                                                             ('Switch_1_input','Switch_1', {'in': 0}),
                                                             ('Switch_input', 'Switch', {'in': 0}),
                                                             ('Switch_input_op', 'Switch_input'),
                                                             ('Switch_1_input_op', 'Switch_1_input')],
                                       update_edge_attrs={})
        pattern_matcher.find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(nodes_with_attrs=[('Merge_output_op', {'kind': 'op', }),
                                                             ('Select_op_data', {'kind': 'data',}),
                                                             ('Select_op', {'kind': 'op', 'op': 'Select'}),
                                                             ('identity', dict(kind='op', op='Identity')),
                                                             ('identity_data', dict(kind='data')),
                                                             ('some_op', dict(kind='op')),
                                                             ('some_op_data', dict(kind='data')),
                                                             ('Switch_1_input', dict(kind='data')),
                                                             ('Switch_input', dict(kind='data')),
                                                             ('Switch_1_input_op', dict(kind='op')),
                                                             ('Switch_input_op', dict(kind='op')),
                                                             ('Switch_2_input', dict(kind='data')),
                                                             ('cond_data', dict(kind='data')),
                                                             ],
                                           edges_with_attrs=[  ('Switch_input_op', 'Switch_input'),
                                                             ('Switch_1_input_op', 'Switch_1_input'),
                                                             ('Switch_input', 'some_op'),
                                                             ('Switch_1_input', 'some_op'),
                                                             ('some_op', 'some_op_data'),
                                                             ('some_op_data', 'identity'),
                                                             ('identity', 'identity_data'),
                                                             ('cond_data', 'Select_op', {'in': 0}),
                                                             ('Switch_2_input', 'Select_op', {'in': 1}),
                                                             ('identity_data', 'Select_op', {'in': 2}),
                                                             ('Select_op', 'Select_op_data'),
                                                             ('Select_op_data', 'Merge_output_op'),
                                                             ])
        (flag, resp) = compare_graphs(graph, graph_ref, 'Merge_output_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
