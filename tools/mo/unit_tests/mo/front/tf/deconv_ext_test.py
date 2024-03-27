# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.tf.deconv_ext import Conv2DBackpropInputFrontExtractor
from unit_tests.utils.extractors import PB, BaseExtractorsTestingClass


class DeconvolutionExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.strides = [1, 2, 3, 4]

    def test_deconv2d_defaults(self):
        node = PB({'pb': PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({"i": self.strides})
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})})
        self.expected = {
            'bias_addable': True,
            'pad': None,  # will be inferred when input shape is known
            'pad_spatial_shape': None,
            'output_spatial_shape': None,
            'output_shape': None,
            'group': None,
        }
        Conv2DBackpropInputFrontExtractor.extract(node)
        self.res = node
        self.expected_call_args = (None, False)
        self.compare()

    def test_deconv2d_nhwc(self):
        node = PB({'pb': PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({"i": self.strides})
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})})

        self.expected = {
            "spatial_dims": [1, 2],
            "channel_dims": [3],
            "batch_dims": [0],
            'stride': np.array(self.strides, dtype=np.int8),
        }

        Conv2DBackpropInputFrontExtractor.extract(node)
        self.res = node
        self.expected_call_args = (None, False)
        self.compare()

    def test_deconv2d_nchw(self):
        node = PB({'pb': PB({'attr': {
            'data_format': PB({
                's': b"NCHW"
            }),
            'strides': PB({
                'list': PB({"i": self.strides})
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})})
        self.expected = {
            "spatial_dims": [2, 3],
            "channel_dims": [1],
            "batch_dims": [0],
            'stride': np.array(self.strides, dtype=np.int8),
        }

        Conv2DBackpropInputFrontExtractor.extract(node)
        self.res = node
        self.expected_call_args = (None, False)
        self.compare()
