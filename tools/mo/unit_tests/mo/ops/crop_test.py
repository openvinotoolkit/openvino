# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.crop import Crop
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph


class TestCropPartialInfer(unittest.TestCase):
    @staticmethod
    def _create_graph_type1():
        nodes_attributes = {'crop_input': {'shape': None, 'value': None, 'kind': 'data'},
                            'crop_node': {'op': 'Crop', 'kind': 'op'},
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
                            'crop_node': {'op': 'Crop', 'kind': 'op'},
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
                            'crop_node': {'op': 'Crop', 'kind': 'op'},
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

        with self.assertRaisesRegex(Error, "axis attribute is missing .*"):
            Crop.infer(crop_node)

    def test_crop_type1_infer_neg2(self):
        graph = self._create_graph_type1()

        crop_node = Node(graph, 'crop_node')
        crop_node['crop_begin'] = int64_array([1, 2, 3])

        with self.assertRaisesRegex(Error, "number of crop_begin.*"):
            Crop.infer(crop_node)

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

        with self.assertRaisesRegex(Error, "Number of axis.*"):
            Crop.infer(crop_node)

    def test_crop_type2_infer_neg2(self):
        graph = self._create_graph_type2()

        crop_node = Node(graph, 'crop_node')
        crop_node['dim'] = None
        crop_node['crop_begin'] = None

        with self.assertRaisesRegex(Error, "Crop node crop_node should have either.*"):
            Crop.infer(crop_node)

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

        with self.assertRaisesRegex(Error, "Not all input shapes were defined.*"):
            Crop.infer(crop_node)

    def test_crop_type3_infer_neg2(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        crop_node['axis'] = None

        with self.assertRaisesRegex(Error, "axis attribute is missing for .*"):
            Crop.infer(crop_node)

    def test_crop_type3_infer_neg3(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        crop_node['offset'] = None

        with self.assertRaisesRegex(Error, "offset attribute is missing.*"):
            Crop.infer(crop_node)

    def test_crop_type3_infer_neg4(self):
        graph = self._create_graph_type3()

        crop_node = Node(graph, 'crop_node')
        crop_input2 = Node(graph, 'crop_input2')
        crop_input2.shape = int64_array([1, 4, 423, 563])

        with self.assertRaisesRegex(Error, "The crop for dimension is out of bounds.*"):
            Crop.infer(crop_node)
