"""
 Copyright (c) 2019 Intel Corporation

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
from argparse import Namespace

import numpy as np

from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
from mo.graph.graph import Node
from mo.utils.cli_parser import get_mean_scale_dictionary, parse_tuple_pairs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_3_data': {'value': None, 'kind': 'data', 'data_type': None},
                    # Placeholders
                    'placeholder_1': {'shape': None, 'type': 'Input', 'kind': 'op', 'op': 'Placeholder'},
                    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2': {'shape': None, 'type': 'Input', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_1': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'pl_2': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_2_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    # ScaleShift layer
                    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
                    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    # Mul op
                    'mul_1': {'type': None, 'kind': 'op', 'op': 'Mul'},
                    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'OpOutput', 'infer': lambda x: None}
                    }


class AddMeanScaleValuesTest(unittest.TestCase):
    def test_add_mean_scale_values_with_data_name(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None, 'data_type': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder', 'name': 'data',
                                        'data_type': None}
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        mean_values = parse_tuple_pairs('(124,117,104)')
        scale_values = parse_tuple_pairs('')

        # input = 'data'
        mean_scale = get_mean_scale_dictionary(mean_values, scale_values, None)
        argv = Namespace(mean_scale_values=mean_scale)
        graph.graph['cmd_params'] = argv
        self.assertEqual(len(graph), 3)
        AddMeanScaleValues().find_and_replace_pattern(graph)
        self.assertEqual(len(graph), 6)

    def test_add_mean_scale_values_without_data_name(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None, 'data_type': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder', 'name': 'data',
                                        'data_type': None}
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        mean_values = parse_tuple_pairs('(124,117,104)')
        scale_values = parse_tuple_pairs('')
        # input = None
        mean_scale = get_mean_scale_dictionary(mean_values, scale_values, None)
        argv = Namespace(mean_scale_values=mean_scale)
        graph.graph['cmd_params'] = argv
        self.assertEqual(len(graph), 3)
        AddMeanScaleValues().find_and_replace_pattern(graph)
        self.assertEqual(len(graph), 6)

    def test_add_mean_scale_values1(self):
        graph = build_graph(nodes_attributes,
                            [('pl_1', 'pl_1_data'), ('pl_2', 'pl_2_data')],
                            {'pl_1_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_2_data': {'shape': np.array([1, 6]), 'infer': None},
                             'pl_1': {'shape': np.array([1, 3, 38, 38])},
                             'pl_2': {'shape': np.array([1, 6])},
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        argv = Namespace(
            mean_scale_values={'pl_1': {'mean': np.array([1., 2., 3.])}, 'pl_2': {'mean': np.array([0., 0., 0.])}})
        graph.graph['cmd_params'] = argv
        graph.graph['cmd_params'] = argv
        AddMeanScaleValues().find_and_replace_pattern(graph)
        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 0, "Found Mul op in graph")

    def test_optimize_scale_and_add_mean_values(self):
        graph = build_graph(
            nodes_attributes,
            [
                ('pl_1', 'pl_1_data')
            ],
            {
                'pl_1_data': {
                    'shape': np.array([1, 3, 38, 38]),
                    'infer': None
                },
                'pl_1': {
                    'shape': np.array([1, 3, 38, 38])
                }
            },
            nodes_with_edges_only=True
        )
        graph.graph['layout'] = 'NCHW'
        argv = Namespace(mean_scale_values={'pl_1': {'scale': np.array([1.]), 'mean': np.array([1., 2., 3.])}})
        graph.graph['cmd_params'] = argv
        AddMeanScaleValues().find_and_replace_pattern(graph)
        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 0, "Found Mul op in graph")

    def test_optimize_mean_and_add_scale_values(self):
        graph = build_graph(
            nodes_attributes,
            [
                ('pl_1', 'pl_1_data')
            ],
            {
                'pl_1_data': {
                    'shape': np.array([1, 3, 38, 38]),
                    'infer': None
                },
                'pl_1': {
                    'shape': np.array([1, 3, 38, 38])
                }
            },
            nodes_with_edges_only=True
        )
        graph.graph['layout'] = 'NCHW'
        argv = Namespace(mean_scale_values={'pl_1': {'scale': np.array([1.43]), 'mean': np.array([0., 0., 0.])}})
        graph.graph['cmd_params'] = argv
        AddMeanScaleValues().find_and_replace_pattern(graph)
        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 0, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 1, "Found Mul op in graph")

    def test_add_mean_scale_values3(self):
        graph = build_graph(nodes_attributes,
                            [('pl_1', 'pl_1_data')],
                            {'pl_1_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_1': {'shape': np.array([1, 3, 38, 38])},
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        argv = Namespace(mean_scale_values=[[np.array([1., 2., 3.]), np.array([1., 2., 3.])]])
        graph.graph['cmd_params'] = argv
        AddMeanScaleValues().find_and_replace_pattern(graph)

        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 1, "Found more than one Nul op in graph")

    def test_add_mean_scale_values_cut_graph(self):
        """
        Test case when user cutted start of the network and specified mean/scale value to the new input node 'node_3'.
        """
        graph = build_graph(nodes_attributes,
                            [('pl_1', 'pl_1_data'),
                             ('pl_2', 'pl_2_data'),
                             ('pl_2_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('pl_1_data', 'node_1'),
                             ('node_3_data', 'node_1'),
                             ],
                            {'pl_1_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_2_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_2': {'initial_node_name': 'node_3', 'shape': np.array([1, 3, 38, 38])},
                             'pl_1': {'shape': np.array([1, 3, 38, 38])},
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        argv = Namespace(
            mean_scale_values={'pl_1': {'mean': np.array([1, 2, 3])}, 'node_3': {'scale': np.array([1, 2, 3])}})
        graph.graph['cmd_params'] = argv
        AddMeanScaleValues().find_and_replace_pattern(graph)

        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "There should be exactly one Add op")
        self.assertEqual(mul_op_cnt, 1, "There should be exactly one Mul op")
        self.assertEqual(Node(graph, 'pl_2').out_node().out_node().op, 'Mul', "The Mul op should be added after pl_2")
        self.assertEqual(Node(graph, 'pl_1').out_node().out_node().op, 'Add', "The Add op should be added after pl_1")
