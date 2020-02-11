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

from extensions.middle.EltwiseInputReshape import EltwiseInputReshape
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate_test import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    # Placeholder layers
    'placeholder_1': {'value': None, 'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Reshape layers
    'reshape_1': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_1_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'reshape_1_const_data': {'kind': 'data', 'value': None, 'shape': None},

    'reshape_2': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_2_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'reshape_2_const_data': {'kind': 'data', 'value': None, 'shape': None},

    # Fake consumes layers
    'consumer_1': {'type': 'Consumer', 'value': None, 'kind': 'op', 'op': 'Consumer'},
    'consumer_2': {'type': 'Consumer', 'value': None, 'kind': 'op', 'op': 'Consumer'},
    'consumer_3': {'type': 'Consumer', 'value': None, 'kind': 'op', 'op': 'Consumer'},

    # Concat
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
}


class EltwiseInputReshapeTest(unittest.TestCase):
    def test1_not_constant(self):
        #        ,-------------->consumer3                 ,------------>consumer3
        #   data---(new_shape1)-->consumer1      =>    data---->Reshape-->consumer1
        #        `-(new_shape2)-->consumer2                 `-->Reshape-->consumer2
        #
        graph = build_graph(nodes_attributes, [
            ('placeholder_1', 'placeholder_1_data'),
            ('placeholder_1_data', 'consumer_1', {'new_shape': int64_array([1, 3, 1, 1])}),
            ('placeholder_1_data', 'consumer_2', {'new_shape': int64_array([1, 1, 3])}),
            ('placeholder_1_data', 'consumer_3'),
            ('consumer_1', 'concat'),
            ('consumer_2', 'concat'),
            ('consumer_3', 'concat'),
        ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3])}}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [
                                    ('placeholder_1', 'placeholder_1_data'),
                                    ('placeholder_1_data', 'reshape_1'),
                                    ('reshape_1_const', 'reshape_1_const_data'),
                                    ('reshape_1_const_data', 'reshape_1'),
                                    ('placeholder_1_data', 'reshape_2'),
                                    ('reshape_2_const', 'reshape_2_const_data'),
                                    ('reshape_2_const_data', 'reshape_2'),
                                    ('placeholder_1_data', 'consumer_3'),
                                    ('reshape_1', 'reshape_1_data'),
                                    ('reshape_2', 'reshape_2_data'),
                                    ('reshape_1_data', 'consumer_1'),
                                    ('reshape_2_data', 'consumer_2'),
                                    ('consumer_1', 'concat'),
                                    ('consumer_2', 'concat'),
                                    ('consumer_3', 'concat'),
                                ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3])},
                                 'reshape_1_const': {'value': int64_array([1, 3, 1, 1]), 'shape': int64_array([4])},
                                 'reshape_1_const_data': {'value': int64_array([1, 3, 1, 1]),
                                                          'shape': int64_array([4])},
                                 'reshape_1_data': {'shape': int64_array([1, 3, 1, 1])},
                                 'reshape_2_const': {'value': int64_array([1, 1, 3]), 'shape': int64_array([3])},
                                 'reshape_2_const_data': {'value': int64_array([1, 1, 3]), 'shape': int64_array([3])},
                                 'reshape_2_data': {'shape': int64_array([1, 1, 3])},
                                 }, nodes_with_edges_only=True)

        pattern = EltwiseInputReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2_not_constant(self):
        #        ,--------------->consumer3                ,----------->consumer3
        #   data---(new_shape1)-->consumer1      =>    data-->Reshape-->consumer1
        #        `-(new_shape1)-->consumer2                         `-->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [
                                ('placeholder_1', 'placeholder_1_data'),
                                ('placeholder_1_data', 'consumer_1', {'new_shape': int64_array([1, 3, 1, 1])}),
                                ('placeholder_1_data', 'consumer_2', {'new_shape': int64_array([1, 3, 1, 1])}),
                                ('placeholder_1_data', 'consumer_3'),
                                ('consumer_1', 'concat'),
                                ('consumer_2', 'concat'),
                                ('consumer_3', 'concat'),
                            ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3])}}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [
                                    ('placeholder_1', 'placeholder_1_data'),
                                    ('placeholder_1_data', 'reshape_1'),
                                    ('reshape_1_const', 'reshape_1_const_data'),
                                    ('reshape_1_const_data', 'reshape_1'),
                                    ('placeholder_1_data', 'consumer_3'),
                                    ('reshape_1', 'reshape_1_data'),
                                    ('reshape_1_data', 'consumer_1'),
                                    ('reshape_1_data', 'consumer_2'),
                                    ('consumer_1', 'concat'),
                                    ('consumer_2', 'concat'),
                                    ('consumer_3', 'concat'),
                                ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3])},
                                 'reshape_1_const': {'value': int64_array([1, 3, 1, 1]), 'shape': int64_array([4])},
                                 'reshape_1_const_data': {'value': int64_array([1, 3, 1, 1]),
                                                          'shape': int64_array([4])},
                                 'reshape_1_data': {'shape': int64_array([1, 3, 1, 1])},
                                 }, nodes_with_edges_only=True)

        pattern = EltwiseInputReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test3_constant(self):
        #        ,--------------->consumer3            data-->consumer3
        #   data---(new_shape1)-->consumer1      =>    data-->consumer1
        #        `-(new_shape2)-->consumer2            data-->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'consumer_1', {'new_shape': int64_array([1, 3, 1, 1])}),
                             ('placeholder_1_data', 'consumer_2', {'new_shape': int64_array([1, 1, 3])}),
                             ('placeholder_1_data', 'consumer_3'),
                             ('consumer_1', 'concat'),
                             ('consumer_2', 'concat'),
                             ('consumer_3', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3]), 'value': np.ones([1, 3])}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'consumer_1'),
                                 ('placeholder_2_data', 'consumer_2'),
                                 ('placeholder_3_data', 'consumer_3'),
                                 ('consumer_1', 'concat'),
                                 ('consumer_2', 'concat'),
                                 ('consumer_3', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3, 1, 1]),
                                                        'value': np.ones([1, 3, 1, 1])},
                                 'placeholder_2_data': {'shape': int64_array([1, 1, 3]), 'value': np.ones([1, 1, 3])},
                                 'placeholder_3_data': {'shape': int64_array([1, 3]), 'value': np.ones([1, 3])},
                                 }, nodes_with_edges_only=True)

        pattern = EltwiseInputReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test4_constant(self):
        #        ,--------------->consumer3                 ,-->consumer3
        #   data---(new_shape1)-->consumer1      =>    data-->consumer1
        #        `-(new_shape2)-->consumer2                 `->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'consumer_1', {'new_shape': int64_array([3, 1, 1])}),
                             ('placeholder_1_data', 'consumer_2', {'new_shape': int64_array([3, 1, 1])}),
                             ('placeholder_1_data', 'consumer_3', {'new_shape': int64_array([3, 1, 1])}),
                             ('consumer_1', 'concat'),
                             ('consumer_2', 'concat'),
                             ('consumer_3', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3]), 'value': np.ones([1, 3])}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'consumer_1'),
                                 ('placeholder_1_data', 'consumer_2'),
                                 ('placeholder_1_data', 'consumer_3'),
                                 ('consumer_1', 'concat'),
                                 ('consumer_2', 'concat'),
                                 ('consumer_3', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': int64_array([3, 1, 1]), 'value': np.ones([3, 1, 1])}
                                 }, nodes_with_edges_only=True)

        pattern = EltwiseInputReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test5_not_constant(self):
        #        ,--------------->consumer3                ,->consumer3
        #   data---(new_shape1)-->consumer1      =>    data----->consumer1
        #        `-(new_shape1)-->consumer2                `-->consumer2
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'consumer_1', {'new_shape': int64_array([1, 3])}),
                             ('placeholder_1_data', 'consumer_2', {'new_shape': int64_array([1, 3])}),
                             ('placeholder_1_data', 'consumer_3'),
                             ('consumer_1', 'concat'),
                             ('consumer_2', 'concat'),
                             ('consumer_3', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': int64_array([1, 3])}}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'consumer_1', {'new_shape': int64_array([1, 3])}),
                                 ('placeholder_1_data', 'consumer_2', {'new_shape': int64_array([1, 3])}),
                                 ('placeholder_1_data', 'consumer_3'),
                                 ('consumer_1', 'concat'),
                                 ('consumer_2', 'concat'),
                                 ('consumer_3', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': int64_array([1, 3])}}, nodes_with_edges_only=True)

        pattern = EltwiseInputReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)
