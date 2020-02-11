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
import numpy as np

from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.extractors.convolutional_component_ext import ConvolutionalComponentFrontExtractor
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.convolution import Convolution
from mo.ops.op import Op


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
        pb += KaldiFrontExtractorTest.generate_matrix([2, 1])
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
        self.assertTrue(np.array_equal(self.test_node.weights, [0, 1]))
        self.assertTrue(np.array_equal(self.test_node.biases, [0, 1]))

