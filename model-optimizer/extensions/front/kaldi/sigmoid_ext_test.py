# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.front.kaldi.sigmoid_ext import SigmoidFrontExtractor
from extensions.ops.activation_ops import Sigmoid
from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.ops.op import Op


class SigmoidFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Sigmoid'] = Sigmoid

    def test_assertion(self):
        self.assertRaises(AttributeError, SigmoidFrontExtractor.extract, None)

    def test_extracted_blobs_add_shift(self):
        SigmoidFrontExtractor.extract(self.test_node)
        self.assertTrue(self.test_node.op, 'Sigmoid')
