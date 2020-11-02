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
from unittest.mock import MagicMock

import numpy as np

from mo.front.common.partial_infer.caffe_fallback import build_net
from mo.utils.unittest.extractors import FakeMultiParam, FakeValue
from mo.utils.unittest.graph import build_graph


class Net:
    def __init__(self, blobs):
        self.blobs = blobs
        self.reshape_blob = MagicMock(return_value=np.array([1, 1, 1, 1]))
        self.reshape = MagicMock(return_value=np.array([1, 1, 1, 1]))
        self.forward = MagicMock(return_value={'top_node': FakeValue(np.array([1, 3, 112, 112]))})


my_mock_net = None


class Caffe:
    def __init__(self):
        self.TEST = 'TEST'

    def Net(self, *args):
        return my_mock_net


class TestCaffeNativePartialInfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.modules['caffe'] = Caffe()
        cls.nodes_attributes = {
            'node_1': {'type': 'Parameter', 'kind': 'op'},
            'node_2': {'type': 'Parameter', 'kind': 'op'},
            'node_3': {'type': 'Identity', 'kind': 'op'},
            'node_4': {'type': 'Identity', 'kind': 'op'},
            'op_output': { 'kind': 'op', 'op': 'Result'}
        }

    def test_build_net_equal_inputs(self):
        global my_mock_net
        my_blobs = {
            'node_1': FakeValue(np.array([1, 3, 227, 227])),
            'node_2': FakeValue(np.array([1, 3, 224, 224]))
        }
        my_mock_net = Net(my_blobs)
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'node_3'),
                                ('node_2', 'node_3'),
                                ('node_3', 'node_4'),
                                ('node_4', 'op_output')
                            ],
                            {
                                'node_4': {'shape': None},
                                'node_1': {'shape': np.array([1, 3, 227, 227])},
                                'node_2': {'shape': np.array([1, 3, 224, 224])},
                                'node_3': {'top': 'top_node'}
                            })
        graph.proto_path = 'path_to_proto'
        graph.caffemodel_path = 'path_to_proto'
        build_net(graph)
        my_mock_net.reshape.assert_not_called()
        my_mock_net.forward.assert_called_once_with()
        self.assertIsNotNone(graph.caffe_net)

    def test_build_net_not_equal_inputs(self):
        global my_mock_net
        input_node_param = {
            'shape': np.array([1, 3, 112, 112]),
            'reshape': MagicMock(return_value=134)
        }
        my_blobs = {
            'node_1': FakeMultiParam(input_node_param),
        }
        my_mock_net = Net(my_blobs)
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'node_3'),
                                ('node_3', 'node_4'),
                                ('node_4', 'op_output')
                            ],
                            {'node_4': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_3': {'top': 'top_node'}
                             },
                            nodes_with_edges_only=True)
        graph.proto_path = 'path_to_proto'
        graph.caffemodel_path = 'path_to_proto'
        build_net(graph)
        my_mock_net.reshape.assert_called_once_with()
        my_mock_net.forward.assert_called_once_with()
        self.assertIsNotNone(graph.caffe_net)
