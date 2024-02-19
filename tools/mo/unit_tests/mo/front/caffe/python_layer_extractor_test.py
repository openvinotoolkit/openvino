# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.caffe.python_layer_extractor import PythonFrontExtractorOp
from openvino.tools.mo.front.extractor import CaffePythonFrontExtractorOp
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


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
