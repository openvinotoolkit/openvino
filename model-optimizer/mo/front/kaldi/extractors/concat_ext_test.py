# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.extractors.concat_ext import ConcatFrontExtractor
from mo.ops.convolution import Convolution
from mo.ops.op import Op


class ConcatFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Concat'] = Convolution

    def test_concat(self):
        ConcatFrontExtractor.extract(self.test_node)
        self.assertEqual(self.test_node.axis, 1)
