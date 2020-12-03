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

from extensions.back.LayoutChangeForGatherND import LayoutChangeForGatherND
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # GatherND
    'gathernd': {'type': 'GatherND', 'kind': 'op', 'op': 'GatherND'},
    'gathernd_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Result layer
    'result': {'type': 'Result', 'kind': 'op', 'op': 'Result'},
    # Transpose layers
    'transpose_1': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'need_shape_inference': True},
    'transpose_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axis_1_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'axis_1_const_data': {'kind': 'data', 'value': None, 'shape': None},
    'transpose_2': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'need_shape_inference': True},
    'transpose_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axis_2_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'axis_2_const_data': {'kind': 'data', 'value': None, 'shape': None},
    'transpose_3': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'need_shape_inference': True},
    'transpose_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axis_3_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'axis_3_const_data': {'kind': 'data', 'value': None, 'shape': None},
}


class LayoutChangeForGatherNDTests(unittest.TestCase):
    def test_tf_all_ports(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'gathernd'),
                             ('placeholder_2_data', 'gathernd'),
                             ('gathernd', 'gathernd_data'),
                             ('gathernd_data', 'result'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},
                             'placeholder_2_data': {'shape': np.array([1, 3, 224, 224])},
                             'gathernd_data': {'shape': np.array([1, 3, 224, 224])},
                             })
        graph.graph['fw'] = 'tf'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_1_data', 'transpose_1'),
                                 ('axis_1_const', 'axis_1_const_data'),
                                 ('axis_1_const_data', 'transpose_1'),
                                 ('transpose_1', 'transpose_1_data'),
                                 ('placeholder_2_data', 'transpose_2'),
                                 ('axis_2_const', 'axis_2_const_data'),
                                 ('axis_2_const_data', 'transpose_2'),
                                 ('transpose_2', 'transpose_2_data'),
                                 ('transpose_1_data', 'gathernd'),
                                 ('transpose_2_data', 'gathernd'),
                                 ('gathernd', 'gathernd_data'),
                                 ('gathernd_data', 'transpose_3'),
                                 ('axis_3_const', 'axis_3_const_data'),
                                 ('axis_3_const_data', 'transpose_3'),
                                 ('transpose_3', 'transpose_3_data'),
                                 ('transpose_3_data', 'result'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},
                                 'placeholder_2_data': {'shape': np.array([1, 3, 224, 224])},
                                 'axis_1_const_data': {'value': int64_array([0, 2, 3, 1])},
                                 'axis_2_const_data': {'value': int64_array([0, 2, 3, 1])},
                                 'gathernd_data': {'shape': np.array([1, 3, 224, 224])},
                                 'axis_3_const_data': {'value': int64_array([0, 3, 1, 2])},
                                 })

        pattern = LayoutChangeForGatherND()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_tf_one_ports(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'gathernd'),
                             ('placeholder_2_data', 'gathernd'),
                             ('gathernd', 'gathernd_data'),
                             ('gathernd_data', 'result'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},
                             'placeholder_2_data': {'shape': np.array([1, 3])},
                             'gathernd_data': {'shape': np.array([1, 3])},
                             })
        graph.graph['fw'] = 'tf'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_1_data', 'transpose_1'),
                                 ('axis_1_const', 'axis_1_const_data'),
                                 ('axis_1_const_data', 'transpose_1'),
                                 ('transpose_1', 'transpose_1_data'),
                                 ('transpose_1_data', 'gathernd'),
                                 ('placeholder_2_data', 'gathernd'),
                                 ('gathernd', 'gathernd_data'),
                                 ('gathernd_data', 'result'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 224, 224])},
                                 'placeholder_2_data': {'shape': np.array([1, 3])},
                                 'axis_1_const_data': {'value': int64_array([0, 2, 3, 1])},
                                 'gathernd_data': {'shape': np.array([1, 3])}
                                 })

        pattern = LayoutChangeForGatherND()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
