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
from unittest.mock import patch

from mo.front.caffe.extractor import check_phase, register_caffe_python_extractor
from mo.front.extractor import CaffePythonFrontExtractorOp
from mo.graph.graph import Node
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_2': {'type': 'Identity', 'kind': 'op'}}


class TestExtractor(unittest.TestCase):
    def test_check_phase_train_phase(self):
        phase_param = {
            'phase': 0
        }

        include_param = {
            'include': [FakeMultiParam(phase_param)]
        }

        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {
                                'node_1': {'pb': FakeMultiParam(include_param)}
                            })

        node = Node(graph, 'node_1')
        res = check_phase(node)
        exp_res = {'phase': 0}
        self.assertEqual(res, exp_res)

    def test_check_phase_test_phase(self):
        phase_param = {
            'phase': 1
        }

        include_param = {
            'include': [FakeMultiParam(phase_param)]
        }

        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {
                                'node_1': {'pb': FakeMultiParam(include_param)}
                            })

        node = Node(graph, 'node_1')
        res = check_phase(node)
        exp_res = {'phase': 1}
        self.assertEqual(res, exp_res)

    def test_check_phase_no_phase(self):
        phase_param = {}

        include_param = {
            'include': [FakeMultiParam(phase_param)]
        }

        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {
                                'node_1': {'pb': FakeMultiParam(include_param)}
                            })

        node = Node(graph, 'node_1')
        res = check_phase(node)
        exp_res = {}
        self.assertEqual(res, exp_res)

    def test_check_phase_no_include(self):
        include_param = {}

        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {
                                'node_1': {'pb': FakeMultiParam(include_param)}
                            })

        node = Node(graph, 'node_1')
        res = check_phase(node)
        exp_res = {}
        self.assertEqual(res, exp_res)

    def test_check_phase_no_pb(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {})

        node = Node(graph, 'node_1')
        res = check_phase(node)
        exp_res = {}
        self.assertEqual(res, exp_res)

    @patch('mo.ops.activation.Activation')
    def test_register_caffe_python_extractor_by_name(self, op_mock):
        op_mock.op = 'TestLayer'
        name = 'myTestLayer'
        register_caffe_python_extractor(op_mock, name)
        self.assertIn(name, CaffePythonFrontExtractorOp.registered_ops)

    @patch('mo.ops.activation.Activation')
    def test_register_caffe_python_extractor_by_op(self, op_mock):
        op_mock.op = 'TestLayer'
        register_caffe_python_extractor(op_mock)
        self.assertIn(op_mock.op, CaffePythonFrontExtractorOp.registered_ops)
