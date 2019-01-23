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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.crop import Crop
from mo.utils.unittest.graph import build_graph


class TestCropPartialInfer(unittest.TestCase):
    @staticmethod
    def _create_graph_type1():
        nodes_attributes = {'crop_input': {'shape': None, 'value': None, 'kind': 'data'},
                            'crop_node': {'type': 'Crop', 'kind': 'op'},
                            'crop_output': {'shape': None, 'value': None, 'kind': 'data'}
                            }
        return build_graph(nodes_attributes,
                           [
                               ('crop_input', 'crop_node'), ('crop_node', 'crop_output')
                           ],
                           {
                               'crop_input': {'shape': int64_array([1, 3, 224, 224])},
                               'crop_node': {'axis': int64_array([2, 3]),
                                             'crop_begin': int64_array([10, 15]),
                                             'crop_end': int64_array([10, 15])
                                             },
                           })

    @staticmethod
    def _create_graph_type2():
        nodes_attributes = {'crop_input': {'shape': None, 'value': None, 'kind': 'data'},
                            'crop_node': {'type': 'Crop', 'kind': 'op'},
                            'crop_output': {'shape': None, 'value': None, 'kind': 'data'}
                            }
        return build_graph(nodes_attributes,
                           [
                               ('crop_input', 'crop_node'), ('crop_node', 'crop_output')
                           ],
                           {
                               'crop_input': {'shape': int64_array([1, 3, 224, 224])},
                               'crop_node': {'axis': int64_array([2, 3]), 'dim': int64_array([100, 150])},
                           })

    @staticmethod
    def _create_graph_type3():
        nodes_attributes = {'crop_input': {'shape': None, 'value': None, 'kind': 'data'},
                            'crop_input2': {'shape': None, 'value': None, 'kind': 'data'},
                            'crop_node': {'type': 'Crop', 'kind': 'op'},
                            'crop_output': {'shape': None, 'value': None, 'kind': 'data'}
                            }
        return build_graph(nodes_attributes,
                           [
                               ('crop_input', 'crop_node'), ('crop_input2', 'crop_node'), ('crop_node', 'crop_output')
                           ],
                           {
                               'crop_input': {'shape': int64_array([1, 3, 224, 224])},
                               'crop_input2': {'shape': int64_array([1, 3, 100, 150])},
                               'crop_node': {'axis': 2, 'offset': int64_array([10, 15])},
                           })

    def test_crop_type1_infer(self):
        graph = self._create_graph_type1()

        crop_node = Node(graph, 'crop_node')
        Crop.infer(crop_node)

        exp_shape = int64_array([1, 3, 204, 194])
        res_shape = graph.node['crop_output']['shape']

        self.assertTrue(np.array_equal(exp_shape, res_shape),
                        'shapes do not match expected: {} and given: {}'.format(exp_shape, res_shape))

    def test_crop_type1_infer_neg1(self):
        graph = self._create_graph_type1()

        crop_node = Node(graph, 'crop_node')
        crop_node['axis'] = None

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)

    def test_crop_type1_infer_neg2(self):
        graph = self._create_graph_type1()

        crop_node = Node(graph, 'crop_node')
        crop_node['crop_begin'] = int64_array([1, 2, 3])

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)

    def test_crop_type2_infer(self):
        graph = self._create_graph_type2()

        crop_node = Node(graph, 'crop_node')
        Crop.infer(crop_node)

        exp_shape = int64_array([1, 3, 100, 150])
        res_shape = graph.node['crop_output']['shape']

        self.assertTrue(np.array_equal(exp_shape, res_shape),
                        'shapes do not match expected: {} and given: {}'.format(exp_shape, res_shape))

    def test_crop_type2_infer_neg1(self):
        graph = self._create_graph_type2()

        crop_node = Node(graph, 'crop_node')
        crop_node['dim'] = int64_array([1, 2, 3])

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)

    def test_crop_type2_infer_neg2(self):
        graph = self._create_graph_type2()

        crop_node = Node(graph, 'crop_node')
        crop_node['dim'] = None
        crop_node['crop_begin'] = None

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)

    def test_crop_type3_infer(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        Crop.infer(crop_node)

        exp_shape = int64_array([1, 3, 100, 150])
        res_shape = graph.node['crop_output']['shape']

        self.assertTrue(np.array_equal(exp_shape, res_shape),
                        'shapes do not match expected: {} and given: {}'.format(exp_shape, res_shape))

    def test_crop_type3_infer_neg1(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        crop_input2 = Node(graph, 'crop_input2')
        crop_input2.shape = None

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)

    def test_crop_type3_infer_neg2(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        crop_node['axis'] = None

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)

    def test_crop_type3_infer_neg3(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        crop_node['offset'] = None

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)

    def test_crop_type3_infer_neg4(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        crop_input2 = Node(graph, 'crop_input2')
        crop_input2.shape = int64_array([1, 4, 423, 563])

        Crop.infer(crop_node)
        self.assertIsNone(crop_node.out_node().shape)
