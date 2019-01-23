"""
 Copyright (c) 2018 Intel Corporation

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
                                       new_nodes_with_attrs=[('maximum', {'kind':'op', 'op': 'Maximum'}),
                                                             ('maximum_data', {'kind': 'data'})],
                                       new_edges_with_attrs=[('maximum', 'maximum_data'),
                                                             ('maximum_data', 'minimum', {'in':1})],
                                       update_nodes_attributes=[('init_1_data', {'value': np.array([0])}),
                                                                ('init_2_data', {'value': np.array([0])}),
                                                                ('add_1_y_data', {'value': np.array(1)}),
                                                                ('add_2_y_data', {'value': np.array(1)}),
                                                                ('loop_cond_data', {'value': None}),
                                                                ('Identity_2_data', {'value': None}),
                                                                ],
                                       update_edge_attrs={('Strided_slice_data', 'minimum',0): {'in': 0}})

        pattern_matcher.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=[('TensorIteratorCondition', {'kind': 'op', 'op': 'TensorIteratorCondition'}),
                              ('loop_cond_data', {'kind': 'data'}),
                              ('identity_data', {'kind': 'data'}),
                              ('StridedSlice', {'kind': 'op', 'op':'StridedSlice'}),
                              ('StridedSlice_data', {'kind': 'data'}),
                              ('Maximum', {'kind': 'op', 'op': 'Maximum'}),
                              ('Maximum_data', {'kind': 'data'}),
                              ('minimum', {'kind': 'op', 'op': 'Minimum'}),
                              ('minimum_data', {'kind': 'data'}),
                              ],
            edges_with_attrs=[('Maximum', 'Maximum_data'),
                              ('Maximum_data', 'minimum'),
                              ('StridedSlice', 'StridedSlice_data'),
                              ('StridedSlice_data', 'TensorIteratorCondition', {'in':0}),
                              ('StridedSlice_data', 'minimum'),
                              ('minimum', 'minimum_data'),
                              ('minimum_data', 'TensorIteratorCondition', {'in':1}),
                              ('TensorIteratorCondition', 'loop_cond_data'),
                              ('TensorIteratorCondition', 'identity_data'),
                              ],
            update_edge_attrs=None,
            new_nodes_with_attrs=[],
            new_edges_with_attrs=[],
            )
        (flag, resp) = compare_graphs(graph, graph_ref, 'loop_cond_data', check_op_attrs=True)
        self.assertTrue(flag, resp)