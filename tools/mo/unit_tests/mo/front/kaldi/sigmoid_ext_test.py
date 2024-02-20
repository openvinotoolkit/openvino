# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.kaldi.sigmoid_ext import SigmoidFrontExtractor
from openvino.tools.mo.ops.activation_ops import Sigmoid
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from openvino.tools.mo.ops.op import Op


class SigmoidFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Sigmoid'] = Sigmoid

    def test_assertion(self):
        self.assertRaises(AttributeError, SigmoidFrontExtractor.extract, None)

    def test_extracted_blobs_add_shift(self):
        SigmoidFrontExtractor.extract(self.test_node)
        self.assertTrue(self.test_node.op, 'Sigmoid')
