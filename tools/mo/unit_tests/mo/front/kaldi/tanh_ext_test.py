# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.kaldi.tanh_component_ext import TanhFrontExtractor
from openvino.tools.mo.ops.activation_ops import Tanh
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from openvino.tools.mo.ops.op import Op


class TanhFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Tanh'] = Tanh

    def test_assertion(self):
        self.assertRaises(AttributeError, TanhFrontExtractor.extract, None)

    def test_extracted_blobs_add_shift(self):
        TanhFrontExtractor.extract(self.test_node)
        self.assertTrue(self.test_node.op, 'Tanh')
