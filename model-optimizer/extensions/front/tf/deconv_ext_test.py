"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.front.tf.deconv_ext import Conv2DBackpropInputFrontExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


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
