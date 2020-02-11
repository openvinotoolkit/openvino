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

from mo.front.caffe.python_layer_extractor import PythonFrontExtractorOp
from mo.front.extractor import CaffePythonFrontExtractorOp
from mo.graph.graph import Node
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode


class FakePythonProtoLayer:
    def __init__(self, params: FakeMultiParam):
        self.type = 'Python'
        self.python_param = params


class FakePythonExtractor:
    @classmethod
    def extract(cls, node: Node):
        return True


class TestPythonLayerExtractor(unittest.TestCase):
    def test_python_extractor_for_op(self):
        module = 'test_module'
        layer = 'test_layer'
        CaffePythonFrontExtractorOp.registered_ops['{}.{}'.format(module, layer)] = \
            lambda node: CaffePythonFrontExtractorOp.parse_param_str(node.pb.python_param.param_str)
        params = FakeMultiParam({
            'module': module,
            'layer': layer,
            'param_str': "'feat_stride': 16"
        })
        ext = PythonFrontExtractorOp.extract(FakeNode(FakePythonProtoLayer(params), None))
        self.assertEqual({'feat_stride': 16}, ext)

    def test_python_extractor_for_extractors(self):
        module = 'test_module'
        layer = 'test_layer'
        CaffePythonFrontExtractorOp.registered_ops['{}.{}'.format(module, layer)] = FakePythonExtractor
        params = FakeMultiParam({
            'module': module,
            'layer': layer,
            'param_str': "'feat_stride': 16"
        })
        self.assertTrue(PythonFrontExtractorOp.extract(FakeNode(FakePythonProtoLayer(params), None)))
