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

import numpy as np

from extensions.middle.TensorIteratorCondition import LoopConditionMatcher
from mo.utils.unittest.graph import build_graph_with_attrs, compare_graphs


class TensorIteratorConditionTests(unittest.TestCase):
    def test(self):
        pattern_matcher = LoopConditionMatcher()
        pattern = pattern_matcher.pattern()

        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'],
                                       new_nodes_with_attrs=[('maximum', {'kind': 'op', 'op': 'Maximum'}),
                                                             ('maximum_data', {'kind': 'data'}),
                                                             ('TensorIteratorInput', {'kind': 'op', 'op': 'TensorIteratorInput'})],
                                       new_edges_with_attrs=[('maximum', 'maximum_data'),
                                                             ('Identity_1_data', 'TensorIteratorInput')],
                                       update_nodes_attributes=[('init_1_data', {'value': np.array([0])}),
                                                                ('init_2_data', {'value': np.array([0])}),
                                                                ('add_1_y_data', {'value': np.array(1)}),
                                                                ('add_2_y_data', {'value': np.array(1)}),
                                                                ('loop_cond_data', {'value': None}),
                                                                ('Identity_2_data', {'value': None}),
                                                                ])

        pattern_matcher.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=[('TensorIteratorCondition', {'kind': 'op', 'op': 'TensorIteratorCondition'}),
                              ('loop_cond_data', {'kind': 'data'}),
                              ('identity_data', {'kind': 'data'}),
                              ('StridedSlice', {'kind': 'op', 'op':'StridedSlice'}),
                              ('StridedSlice_data', {'kind': 'data'}),
                              ('Maximum', {'kind': 'op', 'op': 'Maximum'}),
                              ('Maximum_data', {'kind': 'data'}),
                              ('minimum_data', {'kind': 'data'}),
                              ('TensorIteratorInput', {'kind': 'op', 'op': 'TensorIteratorInput'})
                              ],
            edges_with_attrs=[('Maximum', 'Maximum_data'),
                              ('StridedSlice', 'StridedSlice_data'),
                              ('StridedSlice_data', 'TensorIteratorCondition', {'in':0}),
                              ('minimum_data', 'TensorIteratorCondition', {'in':1}),
                              ('TensorIteratorCondition', 'loop_cond_data'),
                              ('TensorIteratorCondition', 'identity_data'),
                              ('identity_data', 'TensorIteratorInput'),
                              ],
            update_edge_attrs=None,
            new_nodes_with_attrs=[],
            new_edges_with_attrs=[],
            )
        (flag, resp) = compare_graphs(graph, graph_ref, 'loop_cond_data', check_op_attrs=True)
        self.assertTrue(flag, resp)