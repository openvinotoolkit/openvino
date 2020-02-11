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

from mo.front.common.partial_infer.multi_box_detection import multi_box_detection_infer
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'data'},
                    'node_2': {'value': None, 'kind': 'data'},
                    'node_3': {'value': None, 'kind': 'data'},
                    'detection_output_1': {'type': 'DetectionOutput', 'value': None, 'kind': 'op'},
                    'node_4': {'value': None, 'kind': 'data'}
                    }


class TestMultiBoxDetectionInfer(unittest.TestCase):
    def test_prior_box_infer_ideal(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'detection_output_1'),
                             ('node_2', 'detection_output_1'),
                             ('node_3', 'detection_output_1'),
                             ('detection_output_1', 'node_4')],
                            {'node_1': {'shape': np.array([1, 34928])},
                             'node_2': {'shape': np.array([1, 183372])},
                             'node_3': {'shape': np.array([1, 2, 34928])},
                             'detection_output_1': {"background_label_id": "0", "clip": "1",
                                                    "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",
                                                    "confidence_threshold": "0.01", "keep_top_k": "200",
                                                    "nms_threshold": "0.5", "num_classes": "21",
                                                    "share_location": "1", "top_k": "200",
                                                    "variance_encoded_in_target": "0"},
                             'node_4': {'shape': np.array([1, 1, 200, 7])},
                             })

        multi_box_detection_node = Node(graph, 'detection_output_1')
        print(multi_box_detection_node)

        multi_box_detection_infer(multi_box_detection_node)
        exp_shape = np.array([1, 1, 200, 7])
        res_shape = graph.node['node_4']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(multi_box_detection_node.background_label_id, '0')
        self.assertEqual(multi_box_detection_node.clip, '1')
        self.assertEqual(multi_box_detection_node.code_type, 'caffe.PriorBoxParameter.CENTER_SIZE')
        self.assertEqual(multi_box_detection_node.confidence_threshold, '0.01')
        self.assertEqual(multi_box_detection_node.keep_top_k, '200')
        self.assertEqual(multi_box_detection_node.nms_threshold, '0.5')
        self.assertEqual(multi_box_detection_node.num_classes, 21)
        self.assertEqual(multi_box_detection_node.share_location, '1')
        self.assertEqual(multi_box_detection_node.top_k, '200')
        self.assertEqual(multi_box_detection_node.variance_encoded_in_target, '0')

    def test_prior_box_infer_without_top_k(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'detection_output_1'),
                             ('node_2', 'detection_output_1'),
                             ('node_3', 'detection_output_1'),
                             ('detection_output_1', 'node_4')],
                            {'node_1': {'shape': np.array([1, 34928])},
                             'node_2': {'shape': np.array([1, 183372])},
                             'node_3': {'shape': np.array([1, 2, 34928])},
                             'detection_output_1': {"background_label_id": "0", "clip": "1",
                                                    "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",
                                                    "confidence_threshold": "0.01", "keep_top_k": -1,
                                                    "nms_threshold": "0.5", "num_classes": "21",
                                                    "share_location": "1", "top_k": -1,
                                                    "variance_encoded_in_target": "0"},
                             'node_4': {'shape': np.array([1, 1, 69856, 7])},
                             })

        multi_box_detection_node = Node(graph, 'detection_output_1')

        multi_box_detection_infer(multi_box_detection_node)
        exp_shape = np.array([1, 1, 8732, 7])
        res_shape = graph.node['node_4']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertEqual(multi_box_detection_node.background_label_id, '0')
        self.assertEqual(multi_box_detection_node.clip, '1')
        self.assertEqual(multi_box_detection_node.code_type, 'caffe.PriorBoxParameter.CENTER_SIZE')
        self.assertEqual(multi_box_detection_node.confidence_threshold, '0.01')
        self.assertEqual(multi_box_detection_node.keep_top_k, 8732)
        self.assertEqual(multi_box_detection_node.nms_threshold, '0.5')
        self.assertEqual(multi_box_detection_node.num_classes, 21)
        self.assertEqual(multi_box_detection_node.share_location, '1')
        self.assertEqual(multi_box_detection_node.top_k, -1)
        self.assertEqual(multi_box_detection_node.variance_encoded_in_target, '0')

    def test_prior_box_infer_raise_error(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'detection_output_1'),
                             ('node_2', 'detection_output_1'),
                             ('node_3', 'detection_output_1'),
                             ('detection_output_1', 'node_4')],
                            {'node_1': {'shape': np.array([1, 34928])},
                             'node_2': {'shape': np.array([1, 183372])},
                             'node_3': {'shape': np.array([1, 3, 34928])},
                             'detection_output_1': {"background_label_id": "0", "clip": "1",
                                                    "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",
                                                    "confidence_threshold": "0.01", "keep_top_k": -1,
                                                    "nms_threshold": "0.5", "num_classes": "21",
                                                    "share_location": "1", "top_k": -1,
                                                    "variance_encoded_in_target": 0},
                             'node_4': {'shape': np.array([1, 1, 69856, 7])},
                             })

        multi_box_detection_node = Node(graph, 'detection_output_1')

        self.assertIsNone(multi_box_detection_infer(multi_box_detection_node))
