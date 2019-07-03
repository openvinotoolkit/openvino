"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.front.tf.conv_ext import Conv2DFrontExtractor, DepthwiseConv2dNativeFrontExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


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
