# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.tf.conv_ext import Conv2DFrontExtractor, DepthwiseConv2dNativeFrontExtractor
from unit_tests.utils.extractors import PB, BaseExtractorsTestingClass


class ConvExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.strides = [1, 2, 3, 4]
        cls.dilations = [1, 1, 1, 1]

    def test_conv_2d_defaults(self):
        node = PB({'pb': PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({"i": self.strides})
            }),
            'padding': PB({
                's': b'VALID'
            }),
            'dilations': PB({
                'list': PB({"i": [1, 1, 1, 1]})
            })
        }})})
        self.expected = {
            'bias_addable': True,
            'dilation': np.array([1, 1, 1, 1], dtype=np.int8),
            'type': 'Convolution',
            'layout': 'NHWC',
        }
        Conv2DFrontExtractor.extract(node)
        self.res = node
        self.expected_call_args = (None, False)
        self.compare()

    def test_conv2d_nhwc(self):
        node = PB({'pb': PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({"i": self.strides})
            }),
            'padding': PB({
                's': b'VALID'
            }),
            'dilations': PB({
                'list': PB({"i": [1, 1, 1, 1]})
            })
        }})})
        self.expected = {
            # spatial_dims = [1, 2] will be detected in infer function
            "channel_dims": [3],
            "batch_dims": [0],
            "input_feature_channel": 2,
            "output_feature_channel": 3,
            'dilation': np.array([1, 1, 1, 1], dtype=np.int8),
            'stride': np.array(self.strides, dtype=np.int8),
        }
        Conv2DFrontExtractor.extract(node)
        self.res = node
        self.expected_call_args = (None, False)
        self.compare()

    def test_conv2d_nchw(self):
        node = PB({'pb': PB({'attr': {
            'data_format': PB({
                's': b"NCHW"
            }),
            'strides': PB({
                'list': PB({"i": self.strides})
            }),
            'padding': PB({
                's': b'VALID'
            }),
            'dilations': PB({
                'list': PB({"i": [1, 1, 1, 1]})
            })
        }})})
        self.expected = {
            # spatial_dims = [2, 3] will be detected in infer function
            "channel_dims": [1],
            "batch_dims": [0],
            "input_feature_channel": 2,
            "output_feature_channel": 3,
            'dilation': np.array([1, 1, 1, 1], dtype=np.int8),
            'stride': np.array(self.strides, dtype=np.int8),
        }
        Conv2DFrontExtractor.extract(node)
        self.res = node
        self.expected_call_args = (None, False)
        self.compare()

    def test_conv2d_depthwise(self):
        node = PB({'pb': PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({"i": self.strides}),
            }),
            'dilations': PB({
                'list': PB({"i": self.dilations}),
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})})
        self.expected = {
            # spatial_dims = [1, 2] will be detected in infer function
            "channel_dims": [3],
            "batch_dims": [0],
            "input_feature_channel": 2,
            "output_feature_channel": 2,
            'dilation': np.array([1, 1, 1, 1], dtype=np.int8),
            'stride': np.array(self.strides, dtype=np.int8),
        }
        DepthwiseConv2dNativeFrontExtractor.extract(node)
        self.res = node
        self.expected_call_args = (None, True)
        self.compare()
