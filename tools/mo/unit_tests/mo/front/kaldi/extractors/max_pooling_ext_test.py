# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.kaldi.extractors.max_pooling_ext import MaxPoolingComponentFrontExtractor
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.pooling import Pooling
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from unit_tests.mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading


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
            'stride': [1, 1, 1, 2],
            'pool_stride': 4,
            'pad': [[[0, 0], [0, 0], [0, 0], [0, 0]]]
        }
        for attr in val_attrs:
            if isinstance(val_attrs[attr], list):
                self.assertTrue((self.test_node[attr] == val_attrs[attr]).all())
            else:
                self.assertEqual(self.test_node[attr], val_attrs[attr])
