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

from mo.front.common.partial_infer.expand_dims import tf_expand_dims_infer
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'input_1': {'kind': 'data', 'value': None},
                    'input_2': {'kind': 'data', 'value': None},
                    'expand_dims': {'kind': 'op'},
                    'out': {'value': None, 'shape': None, 'kind': 'data'}
                    }


class TestExpandDimsInfer(unittest.TestCase):
    def test_expand_dims_infer_two_inputs(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('input_2', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'input_2': {'shape': np.array([1]), 'value': np.array([1], dtype=np.int32)},
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        exp_shape = np.array([3, 1, 256, 256])
        res_shape = expand_dims_node.out_node().shape
        self.assertEqual(len(exp_shape), len(res_shape))
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_expand_dims_infer_two_inputs_2(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('input_2', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'input_2': {'shape': np.array([1]), 'value': np.array([2], dtype=np.int32)},
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        exp_shape = np.array([3, 256, 1, 256])
        res_shape = expand_dims_node.out_node().shape
        self.assertEqual(len(exp_shape), len(res_shape))
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_expand_dims_infer_two_inputs_3(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('input_2', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'input_2': {'shape': np.array([]), 'value': np.array(3, dtype=np.int32)},
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        exp_shape = np.array([3, 256, 256, 1])
        res_shape = expand_dims_node.out_node().shape
        self.assertEqual(len(exp_shape), len(res_shape))
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_expand_dims_infer_two_inputs_negative(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('input_2', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'input_2': {'shape': np.array([1]), 'value': np.array([2, 3], dtype=np.int32)},
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        self.assertIsNone(expand_dims_node.out_node().shape)

    def test_expand_dims_infer_two_inputs_negative_2(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('input_2', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': None},
                             'input_2': {'shape': np.array([1]), 'value': np.array([2, 3], dtype=np.int32)},
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        self.assertIsNone(expand_dims_node.out_node().shape)

    def test_expand_dims_infer_one_input(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'expand_dims': {'expand_axis': 1}
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        exp_shape = np.array([3, 1, 256, 256])
        res_shape = expand_dims_node.out_node().shape
        self.assertEqual(len(exp_shape), len(res_shape))
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_expand_dims_infer_one_input_2(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'expand_dims': {'expand_axis': 2}
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        exp_shape = np.array([3, 256, 1, 256])
        res_shape = expand_dims_node.out_node().shape
        self.assertEqual(len(exp_shape), len(res_shape))
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_expand_dims_infer_one_input_3(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'expand_dims': {'expand_axis': -1}
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        exp_shape = np.array([3, 256, 256, 1])
        res_shape = expand_dims_node.out_node().shape
        self.assertEqual(len(exp_shape), len(res_shape))
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_expand_dims_infer_one_input_4(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'expand_dims': {'expand_axis': -2}
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        exp_shape = np.array([3, 256, 1, 256])
        res_shape = expand_dims_node.out_node().shape
        self.assertEqual(len(exp_shape), len(res_shape))
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_expand_dims_infer_one_input_negative(self):
        graph = build_graph(nodes_attributes,
                            [('input_1', 'expand_dims'),
                             ('expand_dims', 'out')],
                            {'input_1': {'shape': np.array([3, 256, 256])},
                             'expand_dims': {'expand_axis': None}
                             })

        expand_dims_node = Node(graph, 'expand_dims')

        tf_expand_dims_infer(expand_dims_node)
        self.assertIsNone(expand_dims_node.out_node().shape)
