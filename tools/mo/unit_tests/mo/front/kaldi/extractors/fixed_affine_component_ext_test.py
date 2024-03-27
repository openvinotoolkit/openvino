# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.MatMul import FullyConnected
from openvino.tools.mo.front.kaldi.extractors.fixed_affine_component_ext import FixedAffineComponentFrontExtractor
from openvino.tools.mo.ops.op import Op
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from unit_tests.mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading


class FixedAffineComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['FullyConnected'] = FullyConnected

    @classmethod
    def create_pb_for_test_node(cls):
        pb = b'<LinearParams> ' + KaldiFrontExtractorTest.generate_matrix([10, 10])
        pb += b'<BiasParams> ' + KaldiFrontExtractorTest.generate_vector(10)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)
        FixedAffineComponentFrontExtractor.extract(cls.test_node)

    def test_assertion(self):
        self.assertRaises(AttributeError, FixedAffineComponentFrontExtractor.extract, None)

    def test_attrs(self):
        self.assertEqual(self.test_node['out-size'], 10)

    def test_out_blobs(self):
        self.assertTrue(np.array_equal(self.test_node.weights, range(10 * 10)))
        self.assertTrue(np.array_equal(self.test_node.biases, range(10)))
