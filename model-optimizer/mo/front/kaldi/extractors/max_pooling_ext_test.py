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

from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest

from mo.front.kaldi.extractors.max_pooling_ext import MaxPoolingComponentFrontExtractor
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.op import Op
from mo.ops.pooling import Pooling


class MaxPoolingComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Pooling'] = Pooling

    @classmethod
    def create_pb_for_test_node(cls):
        pb = KaldiFrontExtractorTest.write_tag_with_value('<PoolSize>', 2)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<PoolStep>', 2)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<PoolStride>', 4)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)
        MaxPoolingComponentFrontExtractor.extract(cls.test_node)

    def test_assertion(self):
        self.assertRaises(AttributeError, MaxPoolingComponentFrontExtractor.extract, None)

    def test_attrs(self):
        val_attrs = {
            'window': [1, 1, 1, 2],
            'stride': [1, 1, 2, 2],
            'pool_stride': 4,
            'pad': [[[0, 0], [0, 0], [0, 0], [0, 0]]]
        }
        for attr in val_attrs:
            if isinstance(val_attrs[attr], list):
                self.assertTrue((self.test_node[attr] == val_attrs[attr]).all())
            else:
                self.assertEqual(self.test_node[attr], val_attrs[attr])
