# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.EltwiseInputReshape import normalize_eltwise_inputs
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    # Placeholder layers
    'placeholder_1': {'value': None, 'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'value': None, 'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3': {'value': None, 'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_4_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Reshape layers
    'reshape_1': {'type': 'Unsqueeze', 'value': None, 'kind': 'op', 'op': 'Unsqueeze'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_1_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'reshape_1_const_data': {'kind': 'data', 'value': None, 'shape': None},

    'reshape_2': {'type': 'Unsqueeze', 'value': None, 'kind': 'op', 'op': 'Unsqueeze'},
    'reshape_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_2_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'reshape_2_const_data': {'kind': 'data', 'value': None, 'shape': None},

    # Eltwise consumes layers
    'eltwise_1': {'kind': 'op', 'is_eltwise': True},
    'eltwise_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'eltwise_2': {'kind': 'op', 'is_eltwise': True},
    'eltwise_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    'eltwise_3': {'kind': 'op', 'is_eltwise': True},
    'eltwise_3_data': {'value': None, 'shape': None, 'kind': 'data'},

    'eltwise_4': {'kind': 'op', 'is_eltwise': True},
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
        graph = build_graph(nodes_attributes, [
            ('placeholder_1', 'placeholder_1_data'),
            ('placeholder_1', 'placeholder_2_data'),
            ('placeholder_1', 'placeholder_3_data'),
            ('placeholder_1_data', 'eltwise_1'),
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
                                [
                                    ('placeholder_1', 'placeholder_1_data'),
                                    ('placeholder_1', 'placeholder_2_data'),
                                    ('placeholder_1', 'placeholder_3_data'),
                                    ('placeholder_1_data', 'eltwise_1'),
                                    ('placeholder_2_data', 'reshape_1'),
                                    ('reshape_1_const', 'reshape_1_const_data'),
                                    ('reshape_1_const_data', 'reshape_1'),
                                    ('placeholder_3_data', 'reshape_2'),
                                    ('reshape_2_const', 'reshape_2_const_data'),
                                    ('reshape_2_const_data', 'reshape_2'),
                                    ('reshape_1', 'reshape_1_data'),
                                    ('reshape_2', 'reshape_2_data'),
                                    ('reshape_1_data', 'eltwise_1'),
                                    ('reshape_2_data', 'eltwise_1'),
                                    ('eltwise_1', 'eltwise_1_data')
                                ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'reshape_1_const': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_1_const_data': {'value': int64_array([0]),
                                                          'shape': int64_array([1])},
                                 'reshape_1_data': {'shape': np.array([1, 1, 64, 1])},
                                 'reshape_2_const': {'value': int64_array([0, 1]), 'shape': int64_array([2])},
                                 'reshape_2_const_data': {'value': int64_array([0, 1]),
                                                          'shape': int64_array([2])},
                                 'reshape_2_data': {'shape': np.array([1, 1, 64, 1])},
                                 'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])}
                                 }, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

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
        #   data1(1,3,64,64)---------------------,->Eltwise1->data(1,3,64,64)-----,->Eltwise2->data(1,3,64,64)---,->Eltwise4->data(1,3,64,64)
        #                                       /\                               /\                              /\
        #   data2(64,1)-,- Reshape1(1,1,64,64)--'--------------------------------o-------------------------------'
        #               |                                                        |
        #               |                                                Reshape(1,1,64,1)
        #              \/                                                        |
        #   data3(64,1)----------->Eltwise3->data(64,1)--------------------------'
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_3', 'placeholder_3_data'),
                             ('placeholder_1_data', 'eltwise_1'),
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
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_3', 'placeholder_3_data'),
                                 ('placeholder_1_data', 'eltwise_1'),
                                 ('placeholder_2_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'eltwise_1'),
                                 ('eltwise_1', 'eltwise_1_data'),
                                 ('eltwise_1_data', 'eltwise_2'),
                                 ('placeholder_2_data', 'eltwise_3'),
                                 ('placeholder_3_data', 'eltwise_3'),
                                 ('eltwise_3', 'eltwise_3_data'),
                                 ('eltwise_3_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'eltwise_2'),
                                 ('eltwise_2', 'eltwise_2_data'),
                                 ('eltwise_2_data', 'eltwise_4'),
                                 ('reshape_1_data', 'eltwise_4'),
                                 ('eltwise_4', 'eltwise_4_data'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'placeholder_2_data': {'shape': np.array([64, 1]),
                                                        'value': np.ones([64, 1])},
                                 'placeholder_3_data': {'shape': np.array([64, 1])},
                                 'reshape_1_const': {'value': int64_array([0, 1]), 'shape': int64_array([2])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1]),
                                                          'shape': int64_array([2])},
                                 'reshape_1_data': {'shape': np.array([1, 1, 64, 1])},

                                 'reshape_2_const': {'value': int64_array([0, 1]), 'shape': int64_array([2])},
                                 'reshape_2_const_data': {'value': int64_array([0, 1]),
                                                          'shape': int64_array([2])},
                                 'reshape_2_data': {'shape': np.array([1, 1, 64, 1])},
                                 'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'eltwise_2_data': {'shape': np.array([1, 3, 64, 64])},
                                 'eltwise_3_data': {'shape': np.array([64, 1])},
                                 'eltwise_4_data': {'shape': np.array([1, 3, 64, 64])}
                                 }, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'eltwise_4', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2_not_constant(self):
        #        ,-------------->consumer3                 ,------------>consumer3
        #   data---(new_shape1)-->consumer1      =>    data---->Reshape-->consumer1
        #        `-(new_shape2)-->consumer2                 `-->Reshape-->consumer2
        #
        graph = build_graph(nodes_attributes, [
            ('placeholder_1', 'placeholder_1_data'),
            ('placeholder_1_data', 'eltwise_1'),
            ('placeholder_1_data', 'eltwise_2'),
            ('placeholder_1_data', 'eltwise_3'),
            ('eltwise_1', 'eltwise_1_data'),
            ('eltwise_2', 'eltwise_2_data'),
            ('eltwise_3', 'eltwise_3_data'),
            ('eltwise_1_data', 'concat'),
            ('eltwise_2_data', 'concat'),
            ('eltwise_3_data', 'concat'),
        ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3])},
                             'eltwise_1_data': {'shape': int64_array([1, 1, 1, 3])},
                             'eltwise_2_data': {'shape': int64_array([1, 1, 3])},
                             'eltwise_3_data': {'shape': int64_array([1, 3])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [
                                    ('placeholder_1', 'placeholder_1_data'),
                                    ('placeholder_1_data', 'reshape_1'),
                                    ('reshape_1_const', 'reshape_1_const_data'),
                                    ('reshape_1_const_data', 'reshape_1'),
                                    ('placeholder_1_data', 'reshape_2'),
                                    ('reshape_2_const', 'reshape_2_const_data'),
                                    ('reshape_2_const_data', 'reshape_2'),
                                    ('placeholder_1_data', 'eltwise_3'),
                                    ('reshape_1', 'reshape_1_data'),
                                    ('reshape_2', 'reshape_2_data'),
                                    ('reshape_1_data', 'eltwise_1'),
                                    ('reshape_2_data', 'eltwise_2'),
                                    ('eltwise_1', 'eltwise_1_data'),
                                    ('eltwise_2', 'eltwise_2_data'),
                                    ('eltwise_3', 'eltwise_3_data'),
                                    ('eltwise_1_data', 'concat'),
                                    ('eltwise_2_data', 'concat'),
                                    ('eltwise_3_data', 'concat'),
                                ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3])},
                                 'reshape_1_const': {'value': int64_array([0, 1]), 'shape': int64_array([2])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1]),
                                                          'shape': int64_array([2])},
                                 'reshape_1_data': {'shape': int64_array([1, 1, 1, 3])},
                                 'reshape_2_const': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_2_const_data': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_2_data': {'shape': int64_array([1, 1, 3])},
                                 }, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test3_not_constant(self):
        #        ,--------------->consumer3                ,----------->consumer3
        #   data---(new_shape1)-->consumer1      =>    data-->Reshape-->consumer1
        #        `-(new_shape1)-->consumer2                         `-->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [
                                ('placeholder_1', 'placeholder_1_data'),
                                ('placeholder_1_data', 'eltwise_1'),
                                ('placeholder_1_data', 'eltwise_2'),
                                ('placeholder_1_data', 'eltwise_3'),
                                ('eltwise_1', 'eltwise_1_data'),
                                ('eltwise_2', 'eltwise_2_data'),
                                ('eltwise_3', 'eltwise_3_data'),
                                ('eltwise_1_data', 'concat'),
                                ('eltwise_2_data', 'concat'),
                                ('eltwise_3_data', 'concat'),
                            ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3])},
                             'eltwise_1_data': {'shape': int64_array([1, 1, 1, 3])},
                             'eltwise_2_data': {'shape': int64_array([1, 1, 1, 3])},
                             'eltwise_3_data': {'shape': int64_array([1, 3])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [
                                    ('placeholder_1', 'placeholder_1_data'),
                                    ('placeholder_1_data', 'reshape_1'),
                                    ('reshape_1_const', 'reshape_1_const_data'),
                                    ('reshape_1_const_data', 'reshape_1'),
                                    ('placeholder_1_data', 'eltwise_3'),
                                    ('reshape_1', 'reshape_1_data'),
                                    ('reshape_1_data', 'eltwise_1'),
                                    ('reshape_1_data', 'eltwise_2'),
                                    ('eltwise_1', 'eltwise_1_data'),
                                    ('eltwise_2', 'eltwise_2_data'),
                                    ('eltwise_3', 'eltwise_3_data'),
                                    ('eltwise_1_data', 'concat'),
                                    ('eltwise_2_data', 'concat'),
                                    ('eltwise_3_data', 'concat'),
                                ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3])},
                                 'reshape_1_const': {'value': int64_array([0, 1]), 'shape': int64_array([2])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1]),
                                                          'shape': int64_array([2])},
                                 'reshape_1_data': {'shape': int64_array([1, 1, 1, 3])},
                                 }, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test4_constant(self):
        #        ,--------------->consumer3                 ,------------>consumer3
        #   data---(new_shape1)-->consumer1      =>    data--->reshape1-->consumer1
        #        `-(new_shape2)-->consumer2                 `->reshape2-->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'eltwise_1'),
                             ('placeholder_1_data', 'eltwise_2'),
                             ('placeholder_1_data', 'eltwise_3'),
                             ('eltwise_1', 'eltwise_1_data'),
                             ('eltwise_2', 'eltwise_2_data'),
                             ('eltwise_3', 'eltwise_3_data'),
                             ('eltwise_1_data', 'concat'),
                             ('eltwise_2_data', 'concat'),
                             ('eltwise_3_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3]), 'value': np.ones([1, 3])},
                             'eltwise_1_data': {'shape': int64_array([1, 1, 1, 3])},
                             'eltwise_2_data': {'shape': int64_array([1, 1, 3])},
                             'eltwise_3_data': {'shape': int64_array([1, 3])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'eltwise_1'),
                                 ('placeholder_1_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'eltwise_2'),
                                 ('placeholder_1_data', 'eltwise_3'),
                                 ('eltwise_1', 'eltwise_1_data'),
                                 ('eltwise_2', 'eltwise_2_data'),
                                 ('eltwise_3', 'eltwise_3_data'),
                                 ('eltwise_1_data', 'concat'),
                                 ('eltwise_2_data', 'concat'),
                                 ('eltwise_3_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3]), 'value': np.ones([1, 3])},
                                 'reshape_1_const': {'value': int64_array([0, 1]), 'shape': int64_array([2])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1]),
                                                          'shape': int64_array([2])},
                                 'reshape_1_data': {'shape': int64_array([1, 1, 1, 3])},

                                 'reshape_2_const': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_2_const_data': {'value': int64_array([0]),
                                                          'shape': int64_array([1])},
                                 'reshape_2_data': {'shape': int64_array([1, 1, 3])},
                                 }, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test5_constant(self):
        #        ,-(new_shape)-->consumer3                           ,-->consumer3
        #   data---(new_shape)-->consumer1      =>    data-->reshape---->consumer1
        #        `-(new_shape)-->consumer2                           `-->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'eltwise_1'),
                             ('placeholder_1_data', 'eltwise_2'),
                             ('placeholder_1_data', 'eltwise_3'),
                             ('eltwise_1', 'eltwise_1_data'),
                             ('eltwise_2', 'eltwise_2_data'),
                             ('eltwise_3', 'eltwise_3_data'),
                             ('eltwise_1_data', 'concat'),
                             ('eltwise_2_data', 'concat'),
                             ('eltwise_3_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3]), 'value': np.ones([1, 3])},
                             'eltwise_1_data': {'shape': int64_array([1, 1, 3])},
                             'eltwise_2_data': {'shape': int64_array([1, 1, 3])},
                             'eltwise_3_data': {'shape': int64_array([1, 1, 3])},
                             },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'eltwise_1'),
                                 ('reshape_1_data', 'eltwise_2'),
                                 ('reshape_1_data', 'eltwise_3'),
                                 ('eltwise_1', 'eltwise_1_data'),
                                 ('eltwise_2', 'eltwise_2_data'),
                                 ('eltwise_3', 'eltwise_3_data'),
                                 ('eltwise_1_data', 'concat'),
                                 ('eltwise_2_data', 'concat'),
                                 ('eltwise_3_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3]), 'value': np.ones([1, 3])},
                                 'reshape_1_const': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_1_const_data': {'value': int64_array([0]),
                                                          'shape': int64_array([1])},
                                 'reshape_1_data': {'shape': int64_array([1, 1, 3])},
                                 }, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test6_not_constant(self):
        #        ,--------------->consumer3                ,->consumer3
        #   data---(new_shape1)-->consumer1      =>    data----->consumer1
        #        `-(new_shape1)-->consumer2                `-->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'eltwise_1'),
                             ('placeholder_1_data', 'eltwise_2'),
                             ('placeholder_1_data', 'eltwise_3'),
                             ('eltwise_1', 'eltwise_1_data'),
                             ('eltwise_2', 'eltwise_2_data'),
                             ('eltwise_3', 'eltwise_3_data'),
                             ('eltwise_1_data', 'concat'),
                             ('eltwise_2_data', 'concat'),
                             ('eltwise_3_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3])},
                             'eltwise_1_data': {'shape': int64_array([1, 3])},
                             'eltwise_2_data': {'shape': int64_array([1, 3])},
                             'eltwise_3_data': {'shape': int64_array([1, 3])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'eltwise_1'),
                                 ('placeholder_1_data', 'eltwise_2'),
                                 ('placeholder_1_data', 'eltwise_3'),
                                 ('eltwise_1', 'eltwise_1_data'),
                                 ('eltwise_2', 'eltwise_2_data'),
                                 ('eltwise_3', 'eltwise_3_data'),
                                 ('eltwise_1_data', 'concat'),
                                 ('eltwise_2_data', 'concat'),
                                 ('eltwise_3_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3])}}, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test7_axis1_not_constant(self):
        #
        #   data1(1,3,64,64)----.                                                    data(1,3,64,64)-------.
        #   data2(3,64,1)-------->Eltwise-->data(1,3,64,64)=> data(3,64,1)->Unsqueeze(0)->data(1,3,64,1)-->Eltwise->...
        #   data3(3,1)------'                                    data(3,1)->Unsqueeze(2, 0)->data(1,3,1,1)-'
        #
        graph = build_graph(nodes_attributes, [
            ('placeholder_1', 'placeholder_1_data'),
            ('placeholder_2', 'placeholder_2_data'),
            ('placeholder_3', 'placeholder_3_data'),
            ('placeholder_1_data', 'eltwise_1'),
            ('placeholder_2_data', 'eltwise_1'),
            ('placeholder_3_data', 'eltwise_1'),
            ('eltwise_1', 'eltwise_1_data')
        ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'placeholder_2_data': {'shape': np.array([3, 64, 1])},
                             'placeholder_3_data': {'shape': np.array([3, 1])},
                             'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'eltwise_1' : {'axis': 1}
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [
                                    ('placeholder_1', 'placeholder_1_data'),
                                    ('placeholder_2', 'placeholder_2_data'),
                                    ('placeholder_3', 'placeholder_3_data'),
                                    ('placeholder_1_data', 'eltwise_1'),
                                    ('placeholder_2_data', 'reshape_1'),
                                    ('reshape_1_const', 'reshape_1_const_data'),
                                    ('reshape_1_const_data', 'reshape_1'),
                                    ('placeholder_3_data', 'reshape_2'),
                                    ('reshape_2_const', 'reshape_2_const_data'),
                                    ('reshape_2_const_data', 'reshape_2'),
                                    ('reshape_1', 'reshape_1_data'),
                                    ('reshape_2', 'reshape_2_data'),
                                    ('reshape_1_data', 'eltwise_1'),
                                    ('reshape_2_data', 'eltwise_1'),
                                    ('eltwise_1', 'eltwise_1_data')
                                ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'placeholder_2_data': {'shape': np.array([3, 64, 1])},
                                 'placeholder_3_data': {'shape': np.array([3, 1])},
                                 'reshape_1_const': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_1_const_data': {'value': int64_array([0]),
                                                          'shape': int64_array([1])},
                                 'reshape_1_data': {'shape': np.array([1, 3, 64, 1])},
                                 'reshape_2_const': {'value': int64_array([2, 0]), 'shape': int64_array([2])},
                                 'reshape_2_const_data': {'value': int64_array([2, 0]),
                                                          'shape': int64_array([2])},
                                 'reshape_2_data': {'shape': np.array([1, 3, 1, 1])},
                                 'eltwise_1_data': {'shape': np.array([1, 3, 64, 64])}
                                 }, nodes_with_edges_only=True)

        normalize_eltwise_inputs(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'eltwise_1', check_op_attrs=True)
        self.assertTrue(flag, resp)
