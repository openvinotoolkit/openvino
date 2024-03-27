# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.kaldi.extractors.rescale_ext import RescaleFrontExtractor
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from unit_tests.mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading


class RescaleFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['ScaleShift'] = ScaleShiftOp

    @classmethod
    def create_pb_for_test_node(cls):
        input_shape = cls.test_node.in_node().shape
        pb = cls.write_tag_with_value('<LearnRateCoef>', 0)
        pb += cls.write_tag_with_value('FV', input_shape[1])
        for i in range(input_shape[1]):
            pb += TestKaldiUtilsLoading.pack_value(i, TestKaldiUtilsLoading.float32_fmt)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)
        RescaleFrontExtractor.extract(cls.test_node)

    def test_assertion(self):
        self.assertRaises(AttributeError, RescaleFrontExtractor.extract, None)

    def test_extracted_shapes_add_shift(self):
        weights = self.test_node.weights
        weights_shape = weights.shape[0]
        self.assertEqual(self.test_node.in_node().shape[1], weights_shape)

    def test_extracted_blobs_add_shift(self):
        weights = self.test_node.weights
        self.assertTrue(np.array_equal(weights, range(self.test_node.in_node().shape[1])))
