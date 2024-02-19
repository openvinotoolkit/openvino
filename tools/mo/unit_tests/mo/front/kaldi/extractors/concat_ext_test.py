# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.kaldi.extractors.concat_ext import ConcatFrontExtractor
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.ops.op import Op
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest


class ConcatFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Concat'] = Convolution

    def test_concat(self):
        ConcatFrontExtractor.extract(self.test_node)
        self.assertEqual(self.test_node.axis, 1)
