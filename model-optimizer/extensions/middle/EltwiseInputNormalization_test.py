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

from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from extensions.middle.EltwiseInputReshape import EltwiseInputReshape
from mo.middle.passes.eliminate_test import build_graph
from mo.middle.passes.fusing.fuse_linear_ops_test import compare_graphs

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    # Placeholder layers
    'placeholder_1': {'value': None, 'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_4_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Reshape layers
    'reshape_1': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'reshape_2': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Eltwise consumes layers
    'eltwise_1': {'type': 'Eltwise', 'value': None, 'kind': 'op', 'op': 'Eltwise'},
    'eltwise_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'eltwise_2': {'type': 'Eltwise', 'value': None, 'kind': 'op', 'op': 'Eltwise'},
    'eltwise_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    'eltwise_3': {'type': 'Eltwise', 'value': None, 'kind': 'op', 'op': 'Eltwise'},
    'eltwise_3_data': {'value': None, 'shape': None, 'kind': 'data'},

    'eltwise_4': {'type': 'Eltwise', 'value': None, 'kind': 'op', 'op': 'Eltwise'},
    'eltwise_4_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Concat
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
}


class EltwiseInputNormalizationTest(unittest.TestCase):
    def test1_not_constant(self):
        #
        #   data1(1,3,64,64)----.                                                   data(1,3,64,64)-------.
        #   data2(1,64,1)-------->Eltwise-->data(1,3,64,64)   =>    data(1,64,1)->Reshape->data(1,1,64,1)-->Eltwise->...
        #   data3(64,1)------'                                       data(64,1)->Reshape->data(1,1,64,1)-'
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'eltwise_1'),
                             ('placeholder_2_data', 'eltwise_1'),
                             ('placeholder_3_data', 'eltwise_1'),
                             ('eltwise_1', 'eltwise_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'placeholder_2_data': {'shape': np.array([1, 64, 1])},
                             'placeholder_3_data': {'shape': np.array([64, 1])},
                             'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])}
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'eltwise_1'),
                                 ('placeholder_2_data', 'reshape_1'),
                                 ('placeholder_3_data', 'reshape_2'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_1_data', 'eltwise_1'),
                                 ('reshape_2_data', 'eltwise_1'),
                                 ('eltwise_1', 'eltwise_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'reshape_1': {'dim': np.array([1, 1, 64, 1])},
                                 'reshape_1_data': {'shape': np.array([1, 1, 64, 1])},
                                 'reshape_2': {'dim': np.array([1, 1, 64, 1])},
                                 'reshape_2_data': {'shape': np.array([1, 1, 64, 1])},
                                 'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])}
                                 }, nodes_with_edges_only=True)

        pattern = EltwiseInputNormalize()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'eltwise_1', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mega_hardcore(self):
        #   ORIGINAL GRAPH
        #
        #   data1(1,3,64,64)---,->Eltwise1->data(1,3,64,64)-----,->Eltwise2->data(1,3,64,64)---,->Eltwise4->data(1,3,64,64)
        #                     /\                               /\                             /\
        #   data2(64,1)-----,-'--------------------------------'------------------------------'
        #                  \/                                 /
        #   data3(64,1)----`-->Eltwise3->data(64,1)----------'
        #
        #   REFERENCE GRAPH AFTER TRANSFORMATION
        #
        #   data1(1,3,64,64)---,->Eltwise1->data(1,3,64,64)-----,->Eltwise2->data(1,3,64,64)---,->Eltwise4->data(1,3,64,64)
        #                     /\                               /\                              /\
        #   data2(1,1,64,1)---'--------------------------------'-------------------------------'
        #                                                     /
        #   data4(64,1)-------,                        Reshape(1,1,64,1)
        #                    \/                           |
        #   data3(64,1)------`---->Eltwise3->data(64,1)---'
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'eltwise_1'),
                             ('placeholder_2_data', 'eltwise_1'),
                             ('eltwise_1', 'eltwise_1_data'),
                             ('eltwise_1_data', 'eltwise_2'),
                             ('placeholder_2_data', 'eltwise_3'),
                             ('placeholder_3_data', 'eltwise_3'),
                             ('eltwise_3', 'eltwise_3_data'),
                             ('eltwise_3_data', 'eltwise_2'),
                             ('eltwise_2', 'eltwise_2_data'),
                             ('eltwise_2_data', 'eltwise_4'),
                             ('placeholder_2_data', 'eltwise_4'),
                             ('eltwise_4', 'eltwise_4_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'placeholder_2_data': {'shape': np.array([64, 1]), 'value': np.ones([64, 1])},
                             'placeholder_3_data': {'shape': np.array([64, 1])},
                             'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'eltwise_2_data': {'shape': np.array([1, 3, 64, 64])},
                             'eltwise_3_data': {'shape': np.array([64, 1])},
                             'eltwise_4_data': {'shape': np.array([1, 3, 64, 64])}
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'eltwise_1'),
                             ('placeholder_2_data', 'eltwise_1'),
                             ('eltwise_1', 'eltwise_1_data'),
                             ('eltwise_1_data', 'eltwise_2'),
                             ('placeholder_4_data', 'eltwise_3'),
                             ('placeholder_3_data', 'eltwise_3'),
                             ('eltwise_3', 'eltwise_3_data'),
                             ('eltwise_3_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'eltwise_2'),
                             ('eltwise_2', 'eltwise_2_data'),
                             ('eltwise_2_data', 'eltwise_4'),
                             ('placeholder_2_data', 'eltwise_4'),
                             ('eltwise_4', 'eltwise_4_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'placeholder_2_data': {'shape': np.array([1, 1, 64, 1]), 'value': np.ones([1, 1, 64, 1])},
                             'placeholder_3_data': {'shape': np.array([64, 1])},
                             'placeholder_4_data': {'shape': np.array([64, 1]), 'value': np.ones([64, 1])},
                             'reshape_1': {'dim': np.array([1,1,64,1])},
                             'reshape_1_data': {'shape': np.array([1,1,64,1])},
                             'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'eltwise_2_data': {'shape': np.array([1, 3, 64, 64])},
                             'eltwise_3_data': {'shape': np.array([64, 1])},
                             'eltwise_4_data': {'shape': np.array([1, 3, 64, 64])}
                             }, nodes_with_edges_only=True)

        pattern = EltwiseInputNormalize()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'eltwise_1', check_op_attrs=True)
        self.assertTrue(flag, resp)
