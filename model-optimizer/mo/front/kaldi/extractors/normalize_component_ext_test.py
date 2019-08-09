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

from extensions.ops.normalize import NormalizeOp
from mo.front.kaldi.extractors.normalize_component_ext import NormalizeComponentFrontExtractor
from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.op import Op


class NormalizeComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Normalize'] = NormalizeOp

    @classmethod
    def create_pb_for_test_node(cls):
        pb = KaldiFrontExtractorTest.write_tag_with_value('<InputDim>', 16)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<TargetRms>', 0.5, np.float32)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<AddLogStddev>', 'F', np.string_)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)

    def test_extract(self):
        NormalizeComponentFrontExtractor.extract(self.test_node)
        self.assertEqual(len(self.test_node['embedded_inputs']), 1)
        self.assertListEqual(list(self.test_node['weights']), [2.0])
