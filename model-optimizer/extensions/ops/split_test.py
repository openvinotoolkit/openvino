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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from extensions.ops.split import AttributedSplit, AttributedVariadicSplit
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs


class TestSplitOp(unittest.TestCase):
    nodes = {
        'input': {'kind': 'op'},
        'split_input_data': {'kind': 'data', 'shape': None, 'value': None},
        'split_op': {'kind': 'op', 'axis': None, 'num_splits': None, 'op': 'AttributedSplit'},
        'split_output_0_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_0': {'kind': 'op'},
        'split_output_1_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_1': {'kind': 'op'},
    }
    edges = [
        ('input', 'split_input_data'),
        ('split_input_data', 'split_op'),
        ('split_op', 'split_output_0_data'),
        ('split_output_0_data', 'output_0'),
        ('split_op', 'split_output_1_data'),
        ('split_output_1_data', 'output_1'),
    ]

    def test_split_shape_infer(self):
        #  test configuration
        input_shape = [2, 10]
        input_value = None
        axis = 1
        num_splits = 2
        output_shape = [2, 5]
        output_value = [None, None]

        # action
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array(input_shape),
                                                     'value': input_value},
                                'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                            }
                            )

        split_op = Node(graph, 'split_op')
        AttributedSplit.infer(split_op)

        # reference
        graph_ref = build_graph(self.nodes, self.edges,
                                {
                                    'split_input_data': {'shape': int64_array(input_shape),
                                                         'value': input_value},
                                    'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                                    'split_output_0_data': {'shape': int64_array(output_shape),
                                                            'value': output_value[0]},
                                    'split_output_1_data': {'shape': int64_array(output_shape),
                                                            'value': output_value[1]},
                                }
                                )

        # check
        (flag, resp) = compare_graphs(graph, graph_ref, 'split_input_data')
        self.assertTrue(flag, resp)

    def test_split_value_infer(self):
        #  test configuration
        input_shape = [2, 10]
        input_value = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        axis = 1
        num_splits = 2
        output_shape = [2, 5]
        output_value = [[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]], [[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]]

        # action
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array(input_shape),
                                                     'value': int64_array(input_value)},
                                'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                            }
                            )

        split_op = Node(graph, 'split_op')
        AttributedSplit.infer(split_op)

        # reference
        graph_ref = build_graph(self.nodes, self.edges,
                                {
                                    'split_input_data': {'shape': int64_array(input_shape),
                                                         'value': int64_array(input_value)},
                                    'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                                    'split_output_0_data': {'shape': int64_array(output_shape),
                                                            'value': int64_array(output_value[0])},
                                    'split_output_1_data': {'shape': int64_array(output_shape),
                                                            'value': int64_array(output_value[1])},
                                }
                                )

        # check
        (flag, resp) = compare_graphs(graph, graph_ref, 'split_input_data')
        self.assertTrue(flag, resp)


class TestAttributedVariadicSplitOp(unittest.TestCase):
    nodes = {
        'input': {'kind': 'op'},
        'split_input_data': {'kind': 'data', 'shape': None, 'value': None},
        'split_op': {'kind': 'op', 'axis': None, 'split_lengths': None, 'op': 'AttributedVariadicSplit'},
        'split_output_0_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_0': {'kind': 'op'},
        'split_output_1_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_1': {'kind': 'op'},
        'split_output_2_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_2': {'kind': 'op'},
    }
    edges = [
        ('input', 'split_input_data'),
        ('split_input_data', 'split_op'),
        ('split_op', 'split_output_0_data'),
        ('split_output_0_data', 'output_0'),
        ('split_op', 'split_output_1_data'),
        ('split_output_1_data', 'output_1'),
        ('split_op', 'split_output_2_data'),
        ('split_output_2_data', 'output_2'),
    ]

    def test_splitv_zero(self):
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([2, 13, 10, 0]),
                                             'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        AttributedVariadicSplit.infer(node)

        self.assertTrue(len(node.out_edges()) == 3)
        self.assertTrue(np.all(node.split_lengths == np.array([2, 13, 10])))

    def test_splitv_zero_not_last(self):
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([2, 13, 0, 10]),
                                             'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')

        # extractor should do it
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)
        node.out_port(2).get_connection().set_source(node.out_port(3))

        AttributedVariadicSplit.infer(node)

        self.assertTrue(node.out_port(3).disconnected())
        self.assertTrue(np.all(node.split_lengths == np.array([2, 13, 10])))

    def test_splitv_2_zero_not_last(self):
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([2, 13, 0, 0, 10]),
                                             'out_ports_count': 5},
                            }
                            )
        node = Node(graph, 'split_op')

        # extractor should do it
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)
        node.out_port(2).get_connection().set_source(node.out_port(4))

        AttributedVariadicSplit.infer(node)

        self.assertTrue(node.out_port(4).disconnected())
        self.assertTrue(node.out_port(3).disconnected())
        self.assertTrue(np.all(node.split_lengths == np.array([2, 13, 10])))
