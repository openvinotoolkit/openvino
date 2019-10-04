"""
 Copyright (c) 2019 Intel Corporation

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

import numpy as np

from extensions.ops.pnorm import PNormOp
from mo.front.kaldi.extractors.pnorm_component_ext import PNormComponentFrontExtractor
from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.op import Op


class PNormComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['pnorm'] = PNormOp

    @classmethod
    def create_pb_for_test_node(cls):
        pb = KaldiFrontExtractorTest.write_tag_with_value('<InputDim>', 3500)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<OutputDim>', 350)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<P>', 2, np.float32)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)

    def test_extract(self):
        PNormComponentFrontExtractor.extract(self.test_node)
        self.assertEqual(self.test_node['p'], 2)
        self.assertEqual(self.test_node['group'], 10)
