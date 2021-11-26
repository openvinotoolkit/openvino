# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.kaldi.extractors.convolutional_component_ext import ConvolutionalComponentFrontExtractor
from mo.ops.convolution import Convolution
from mo.ops.op import Op
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from unit_tests.mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading


class ConvolutionalComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Convolution'] = Convolution

    @classmethod
    def create_pb_for_test_node(cls):
        pb = KaldiFrontExtractorTest.write_tag_with_value('<PatchDim>', 2)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<PatchStep>', 2)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<PatchStride>', 4)
        pb += KaldiFrontExtractorTest.generate_learn_info()
        pb += b'<Filters> '
        pb += KaldiFrontExtractorTest.generate_matrix([2, 4])
        pb += b'<Bias> '
        pb += KaldiFrontExtractorTest.generate_vector(2)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)
        ConvolutionalComponentFrontExtractor.extract(cls.test_node)

    def test_assertion(self):
        self.assertRaises(AttributeError, ConvolutionalComponentFrontExtractor.extract, None)

    def test_attrs(self):
        val_attrs = {
            'kernel': [1, 1, 1, 2],
            'stride': [1, 1, 1, 2],
            'pad': [[[0, 0], [0, 0], [0, 0], [0, 0]]],
            'output': 2,
            'patch_stride': 4,
            'spatial_dims': [2, 3],
            'channel_dims': [1],
            'batch_dims': [0],
            'dilation': [1, 1, 1, 1]
        }
        for attr in val_attrs:
            if isinstance(val_attrs[attr], list):
                self.assertTrue((self.test_node[attr] == val_attrs[attr]).all())
            else:
                self.assertEqual(self.test_node[attr], val_attrs[attr])

    def test_convolution_blobs(self):
        self.assertTrue(np.array_equal(self.test_node.weights, [0, 1, 2, 3, 4, 5, 6, 7]))
        self.assertTrue(np.array_equal(self.test_node.biases, [0, 1]))

