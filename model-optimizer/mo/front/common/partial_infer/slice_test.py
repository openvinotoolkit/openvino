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

from mo.front.common.partial_infer.slice import tf_strided_slice_infer, convert_negative_indices, mxnet_slice_axis_infer
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'data'},
                    'Slice_node': {'type': 'Slice', 'kind': 'op'},
                    'node_2': {'value': None, 'kind': 'data'},
                    'node_3': {'value': None, 'kind': 'data'},
                    'node_4': {'value': None, 'kind': 'data'},
                    # StridedSlice node with attrs
                    'sslice_input': {'value': None, 'shape': None, 'kind': 'data'},
                    'sslice_1': {'type': 'StridedSlice', 'value': None, 'kind': 'op', 'op': 'StridedSlice'},
                    'sslice_begin_1': {'value': None, 'shape': None, 'kind': 'data'},
                    'sslice_end_1': {'value': None, 'shape': None, 'kind': 'data'},
                    'sslice_stride_1': {'value': None, 'shape': None, 'kind': 'data'},
                    'sslice_data_1': {'value': None, 'shape': None, 'kind': 'data'},
                    # TF slice
                    'tf_slice_input': {'value': None, 'shape': None, 'kind': 'data'},
                    'tf_slice_begin': {'value': None, 'shape': None, 'kind': 'data'},
                    'tf_slice_size': {'value': None, 'shape': None, 'kind': 'data'},
                    'tf_slice': {'kind': 'op'},
                    'tf_slice_output': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    'op_output_1': {'kind': 'op', 'op': 'Result'},
                    'op_output_2': {'kind': 'op', 'op': 'Result'}
                    }

tf_slice_edges = [('tf_slice_input', 'tf_slice'), ('tf_slice_begin', 'tf_slice'), ('tf_slice_size', 'tf_slice'),
                  ('tf_slice', 'tf_slice_output')]


class TestTFStridedSliceInfer(unittest.TestCase):
    def build_test_graph2(self):
        return build_graph(nodes_attributes,
                           [('sslice_input', 'sslice_1'),
                            ('sslice_begin_1', 'sslice_1'),
                            ('sslice_end_1', 'sslice_1'),
                            ('sslice_stride_1', 'sslice_1'),
                            ('sslice_1', 'sslice_data_1'),
                            ('sslice_data_1', 'op_output')
                            ],
                           {
                            'sslice_input': {'value': np.array([1, 34, 34, 62]),
                                             'shape': np.array([3])},
                            'sslice_begin_1': {'value': np.array([0]), 'shape': np.array([1])},
                            'sslice_end_1': {'value': np.array([4]), 'shape': np.array([1])},
                            'sslice_stride_1': {'value': np.array([1]), 'shape': np.array([1])},
                            'sslice_1': {'shrink_axis_mask': [0], 'ellipsis_mask': [0], 'new_axis_mask': [0],
                                         'begin_mask': [1], 'end_mask': [1]},
                            })

    def build_test_graph(self):
        return build_graph(nodes_attributes,
                           [('sslice_input', 'sslice_1'),
                            ('sslice_begin_1', 'sslice_1'),
                            ('sslice_end_1', 'sslice_1'),
                            ('sslice_stride_1', 'sslice_1'),
                            ('sslice_1', 'sslice_data_1'),
                            ('sslice_data_1', 'op_output')
                            ],
                           {
                            'sslice_input': {'value': None, 'shape': np.array([1, 35, 35, 3])},
                            'sslice_begin_1': {'value': np.array([0, 0, 0, 0]), 'shape': np.array([4])},
                            'sslice_end_1': {'value': np.array([1, 34, 30, 2]), 'shape': np.array([4])},
                            'sslice_stride_1': {'value': np.array([1, 1, 1, 1]),
                                                'shape': np.array([4])},
                            'sslice_1': {'shrink_axis_mask': [0], 'ellipsis_mask': [0], 'new_axis_mask': [0],
                                         'begin_mask': [1], 'end_mask': [1]},
                            })

    def build_test_graph_dim_beg(self):
        return build_graph(nodes_attributes,
                           [('sslice_input', 'sslice_1'),
                            ('sslice_begin_1', 'sslice_1'),
                            ('sslice_end_1', 'sslice_1'),
                            ('sslice_stride_1', 'sslice_1'),
                            ('sslice_1', 'sslice_data_1'),
                            ('sslice_data_1', 'op_output')
                            ],
                           {
                            'sslice_input': {'value': np.array([[1, 34, 34, 62]]),
                                             'shape': np.array([1, 4])},
                            'sslice_begin_1': {'value': np.array([0]), 'shape': np.array([1])},
                            'sslice_end_1': {'value': np.array([4]), 'shape': np.array([1])},
                            'sslice_stride_1': {'value': np.array([1]), 'shape': np.array([1])},
                            'sslice_1': {'shrink_axis_mask': [0], 'ellipsis_mask': [0], 'new_axis_mask': [0],
                                         'begin_mask': [1], 'end_mask': [1]},
                            })

    def test_slice_infer_1(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([1, 34, 30, 2])), 'Wrong output shape detected')

    def test_slice_infer_2(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.end_mask = [1, 0, 0, 1]  # 6
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([1, 35, 35, 2])), 'Wrong output shape detected')

    def test_slice_infer_3(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.in_node(1).value = np.array([0, 10, 10, 0])
        node.end_mask = [1, 0, 0, 1]  # 6
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([1, 25, 25, 2])), 'Wrong output shape detected')

    def test_slice_infer_4(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.in_node(1).value = np.array([0, 10, 10, 0])
        node.begin_mask = [1, 0, 0, 1]  # 6
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([1, 34, 30, 2])), 'Wrong output shape detected')

    def test_slice_infer_5(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.in_node(1).value = np.array([0, 10, 10, 0])
        node.begin_mask = [0, 0, 0, 0]  # 15
        node.end_mask = [0, 0, 0, 0]  # 15
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([1, 35, 35, 3])), 'Wrong output shape detected')

    def test_slice_infer_6(self):
        graph = self.build_test_graph2()
        node = Node(graph, 'sslice_1')
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([4])), 'Wrong output shape detected')
        self.assertTrue(np.array_equal(node.out_node().value, np.array([1, 34, 34, 62])), 'Wrong output value detected')

    def test_slice_infer_7(self):
        graph = self.build_test_graph2()
        node = Node(graph, 'sslice_1')
        node.in_node(1).value = np.array([1])
        node.in_node(2).value = np.array([3])
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([2])), 'Wrong output shape detected')
        self.assertTrue(np.array_equal(node.out_node().value, np.array([34, 34])), 'Wrong output value detected')

    def test_slice_infer_8(self):
        graph = self.build_test_graph2()
        node = Node(graph, 'sslice_1')
        node.new_axis_mask = [1]
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([1, 4])), 'Wrong output shape detected')
        self.assertTrue(np.array_equal(node.out_node().value, np.array([[1, 34, 34, 62]])),
                        'Wrong output value detected')

    def test_slice_infer_9(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.begin_mask = [0, 0, 0, 0]  # 15
        node.end_mask = [0, 0, 0, 0]  # 15
        node.shrink_axis_mask = [1]
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([35, 35, 3])), 'Wrong output shape detected')

    def test_slice_infer_10(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.begin_mask = [0, 0, 0, 0]  # 15
        node.end_mask = [0, 0, 0, 0]  # 15
        node.shrink_axis_mask = [1, 0, 0, 0]
        node.new_axis_mask = [0, 0, 0, 1]  # 8
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([35, 35, 1, 3])), 'Wrong output shape detected')

    def test_slice_infer_11(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.begin_mask = [0, 0, 0, 0]  # 15
        node.end_mask = [0, 0, 0, 0]  # 15
        node.shrink_axis_mask = [1, 0, 1, 0]  # 5
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([35, 3])), 'Wrong output shape detected')

    def test_slice_infer_12(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        node.begin_mask = [0, 0, 0, 0]  # 15
        node.end_mask = [0, 0, 0, 0]  # 15
        node.shrink_axis_mask = [1, 1, 1, 0]  # 7
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([3])), 'Wrong output shape detected')

    def test_slice_infer_13(self):
        graph = self.build_test_graph2()
        node = Node(graph, 'sslice_1')
        node.in_node(1).value = np.array([1])
        node.shrink_axis_mask = [1]
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([])), 'Wrong output shape detected')
        self.assertTrue(np.array_equal(node.out_node().value, np.array(34)), 'Wrong output shape detected')

    def test_slice_infer_14(self):
        graph = self.build_test_graph2()
        node = Node(graph, 'sslice_1')
        node.in_node(3).value = np.array([-1])
        node.end_mask = [0]
        node.begin_mask = [0]
        node.in_node(0).shape = [4]
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([4])), 'Wrong output shape detected')
        print(node.out_node().value)
        self.assertTrue(np.array_equal(node.out_node().value, np.array([62, 34, 34, 1])), 'Wrong output shape detected')

    def test_slice_infer_dim_beg(self):
        graph = self.build_test_graph_dim_beg()
        node = Node(graph, 'sslice_1')
        node.shrink_axis_mask = [1]
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([4])), 'Wrong output shape detected')
        self.assertTrue(np.array_equal(node.out_node().value, np.array([1, 34, 34, 62])), 'Wrong output shape detected')

    def test_slice_infer_neg_end(self):
        graph = self.build_test_graph()
        node = Node(graph, 'sslice_1')
        end_node = Node(graph, 'sslice_end_1')
        end_node.value = np.array([1, -1, -5, -1])
        tf_strided_slice_infer(node)
        self.assertTrue(np.array_equal(node.out_node().shape, np.array([1, 34, 30, 2])), 'Wrong output shape detected')
        self.assertTrue(np.array_equal(end_node.value, np.array([1, -1, -5, -1])), 'Negative values in end were converted to positive')


class TestConvertNegativeIndices(unittest.TestCase):
    def test_convert_negative_indices(self):
        dimensions = np.array([3, 4, 8, 10])
        indices = np.array([2, 0, -3, -4])
        convert_negative_indices(indices, dimensions)
        self.assertTrue(np.array_equal(indices, np.array([2, 0, 5, 6])), 'Wrong dimension indices')


class TestMXNetSliceAxisInfer(unittest.TestCase):
    def test_slice_axis_infer_layer(self):
        graph = build_graph(
            {'node_1': {'name': 'data', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'slice_axis_node': {'name': 'slice_axis_node', 'type': 'sigmoid', 'value': None,
                                 'kind': 'op', 'op': 'slice_axis', },
             'node_3': {'name': 'node_3', 'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [
                ('node_1', 'slice_axis_node'),
                ('slice_axis_node', 'node_3'),
            ],
            {
                'node_1': {'shape': np.array([1, 1024, 19, 19])},
                'slice_axis_node': {'axis': 1, 'offset': 10, 'dim': 25},
            })

        slice_axis_node = Node(graph, 'slice_axis_node')
        mxnet_slice_axis_infer(slice_axis_node)
        res_shape = [1, 15, 19, 19]
        for i in range(0, len(graph.node['node_3']['shape'])):
            self.assertEqual(graph.node['node_3']['shape'][i], res_shape[i])
