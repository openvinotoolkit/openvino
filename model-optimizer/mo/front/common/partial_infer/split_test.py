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

from mo.front.common.partial_infer.split import tf_split_infer, tf_unpack_infer, tf_split_v_infer, split
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, build_graph_with_edge_attrs


class TestTFSplitInfer(unittest.TestCase):
    graph = None

    def setUp(self):
        self.graph = build_graph({'split_dim': {'value': None, 'kind': 'data'},
                                  'data_to_split': {'value': None, 'shape': None, 'kind': 'data'},
                                  'split_node': {'kind': 'op', 'op': 'Split', 'num_split': 3, 'axis': None},
                                  'out_data_1': {'value': None, 'shape': None, 'kind': 'data'},
                                  'out_data_2': {'value': None, 'shape': None, 'kind': 'data'},
                                  'out_data_3': {'value': None, 'shape': None, 'kind': 'data'},
                                  },
                                 [('split_dim', 'split_node'),
                                  ('data_to_split', 'split_node'),
                                  ('split_node', 'out_data_1'),
                                  ('split_node', 'out_data_2'),
                                  ('split_node', 'out_data_3'),
                                  ])

    def test_tf_split_infer(self):
        split_node = Node(self.graph, 'split_node')
        self.graph.node['split_dim']['value'] = np.array(1)
        self.graph.node['data_to_split']['shape'] = int64_array([2, 12, 25, 30])

        tf_split_infer(split_node)
        exp_shape = int64_array([2, 4, 25, 30])
        for out_node in split_node.out_nodes().values():
            self.assertTrue(np.all(exp_shape == out_node.shape))
        self.assertEqual(1, split_node.input_port)

    def test_tf_split_infer_negative_index(self):
        split_node = Node(self.graph, 'split_node')
        self.graph.node['split_dim']['value'] = np.array(-3)
        self.graph.node['data_to_split']['shape'] = int64_array([2, 12, 25, 30])

        tf_split_infer(split_node)
        exp_shape = int64_array([2, 4, 25, 30])
        for out_node in split_node.out_nodes().values():
            self.assertTrue(np.all(exp_shape == out_node.shape))
        self.assertEqual(1, split_node.input_port)

    def test_tf_split_infer_unknown_index(self):
        split_node = Node(self.graph, 'split_node')
        self.graph.node['data_to_split']['shape'] = int64_array([2, 12, 25, 30])

        tf_split_infer(split_node)
        for out_node in split_node.out_nodes().values():
            self.assertIsNone(out_node.shape)

    def test_tf_split_infer_input_shape_is_None(self):
        split_node = Node(self.graph, 'split_node')
        self.graph.node['split_dim']['value'] = np.array(1)

        tf_split_infer(split_node)
        for out_node in split_node.out_nodes().values():
            self.assertIsNone(out_node.shape)

    def test_tf_split_infer_wrong_num_split(self):
        split_node = Node(self.graph, 'split_node')
        self.graph.node['split_dim']['value'] = np.array(0)
        self.graph.node['data_to_split']['shape'] = int64_array([2, 12, 25, 30])

        tf_split_infer(split_node)
        for out_node in split_node.out_nodes().values():
            self.assertIsNone(out_node.shape)


class TestTFSplitVInfer(unittest.TestCase):
    graph = None

    def setUp(self):
        self.graph = build_graph({'data_to_split': {'value': None, 'shape': None, 'kind': 'data'},
                                  'size_splits': {'value': [3, 5, 4], 'kind': 'data'},
                                  'split_dim': {'value': None, 'kind': 'data'},
                                  'split_node': {'kind': 'op', 'op': 'Split', 'axis': None},
                                  'out_data_1': {'value': None, 'shape': None, 'kind': 'data'},
                                  'out_data_2': {'value': None, 'shape': None, 'kind': 'data'},
                                  'out_data_3': {'value': None, 'shape': None, 'kind': 'data'},
                                  },
                                 [('data_to_split', 'split_node'),
                                  ('size_splits', 'split_node'),
                                  ('split_dim', 'split_node'),
                                  ('split_node', 'out_data_1'),
                                  ('split_node', 'out_data_2'),
                                  ('split_node', 'out_data_3'),
                                  ])

    def test_tf_split_infer_three_inputs(self):
        split_node = Node(self.graph, 'split_node')
        self.graph.node['split_dim']['value'] = np.array(1)
        self.graph.node['data_to_split']['shape'] = int64_array([2, 12, 25, 30])

        tf_split_v_infer(split_node)
        exp_shape = [int64_array([2, 3, 25, 30]), int64_array([2, 5, 25, 30]), int64_array([2, 4, 25, 30])]
        for ind, out_node in split_node.out_nodes().items():
            self.assertTrue(np.all(exp_shape[ind] == out_node.shape))


class TestTFUnpack(unittest.TestCase):
    graph = None

    def setUp(self):
        self.graph = build_graph({'data_to_split': {'value': None, 'shape': None, 'kind': 'data'},
                                  'unpack': {'kind': 'op', 'op': 'Split', 'num_split': 3, 'axis': None},
                                  'out_data_1': {'value': None, 'shape': None, 'kind': 'data'},
                                  'out_data_2': {'value': None, 'shape': None, 'kind': 'data'},
                                  'out_data_3': {'value': None, 'shape': None, 'kind': 'data'},
                                  'out_data_4': {'value': None, 'shape': None, 'kind': 'data'},
                                  },
                                 [('data_to_split', 'unpack'),
                                  ('unpack', 'out_data_1'),
                                  ('unpack', 'out_data_2'),
                                  ('unpack', 'out_data_3'),
                                  ])

    def test_tf_unpack_infer(self):
        unpack_node = Node(self.graph, 'unpack')
        self.graph.node['unpack']['axis'] = np.array(1)
        self.graph.node['data_to_split']['shape'] = int64_array([2, 3, 25, 30])

        tf_unpack_infer(unpack_node)
        exp_shape = int64_array([2, 1, 25, 30])
        for out_node in unpack_node.out_nodes().values():
            self.assertTrue(np.all(exp_shape == out_node.shape))

    def test_tf_unpack_infer_default_number_of_pieces(self):
        unpack_node = Node(self.graph, 'unpack')
        self.graph.node['unpack']['axis'] = np.array(1)
        self.graph.node['unpack']['num_split'] = None
        self.graph.node['data_to_split']['shape'] = int64_array([2, 3, 25, 30])

        tf_unpack_infer(unpack_node)
        exp_shape = int64_array([2, 1, 25, 30])
        for out_node in unpack_node.out_nodes().values():
            self.assertTrue(np.all(exp_shape == out_node.shape))

    def test_tf_unpack_infer_not_supported(self):
        # the case when the size of the dimension being unpacked is not equal to number of pieces is not supported
        unpack_node = Node(self.graph, 'unpack')
        self.graph.node['unpack']['axis'] = np.array(1)
        self.graph.node['data_to_split']['shape'] = int64_array([2, 6, 25, 30])

        tf_unpack_infer(unpack_node)
        for out_node in unpack_node.out_nodes().values():
            self.assertIsNone(out_node.shape)


class TestSplitFunc(unittest.TestCase):
    graph = None

    def setUp(self):
        self.graph = build_graph_with_edge_attrs(
            {'data_to_split': {'value': None, 'shape': int64_array([2, 12, 25, 44]), 'kind': 'data'},
             'split_node': {'kind': 'op', 'op': 'Split', 'axis': None},
             'out_data_2': {'value': None, 'shape': None, 'kind': 'data'},
             'out_data_5': {'value': None, 'shape': None, 'kind': 'data'},
             'out_data_7': {'value': None, 'shape': None, 'kind': 'data'},
             },
            [('data_to_split', 'split_node', {'in': 0}),
             ('split_node', 'out_data_2', {'out': 2}),
             ('split_node', 'out_data_5', {'out': 5}),
             ('split_node', 'out_data_7', {'out': 7}),
             ])

    def test_split_non_sequential_output_port(self):
        split(Node(self.graph, 'data_to_split'), Node(self.graph, 'split_node'), -1, [3, 2, 7, 5, 6, 4, 9, 8])
        self.assertTrue(np.all(Node(self.graph, 'out_data_2').shape == [2, 12, 25, 7]))
        self.assertTrue(np.all(Node(self.graph, 'out_data_5').shape == [2, 12, 25, 4]))
        self.assertTrue(np.all(Node(self.graph, 'out_data_7').shape == [2, 12, 25, 8]))

    def test_split_value_infer_non_sequential_output_port(self):
        data_node = Node(self.graph, 'data_to_split')
        value = np.array(range(2 * 12 * 25 * 44)).reshape(data_node.shape)
        data_node.value = value.copy()
        split(data_node, Node(self.graph, 'split_node'), -1, [3, 2, 7, 5, 6, 4, 9, 8])
        self.assertTrue(np.all(Node(self.graph, 'out_data_2').shape == [2, 12, 25, 7]))
        self.assertTrue(np.all(Node(self.graph, 'out_data_5').shape == [2, 12, 25, 4]))
        self.assertTrue(np.all(Node(self.graph, 'out_data_7').shape == [2, 12, 25, 8]))

        self.assertTrue(np.all(Node(self.graph, 'out_data_2').value == value[:, :, :, 5:12]))
        self.assertTrue(np.all(Node(self.graph, 'out_data_5').value == value[:, :, :, 23:27]))
        self.assertTrue(np.all(Node(self.graph, 'out_data_7').value == value[:, :, :, 36:]))
