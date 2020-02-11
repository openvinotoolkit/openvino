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
import unittest

from google.protobuf import text_format

from mo.front.caffe.custom_layers_mapping import proto_extractor
from mo.front.caffe.proto import caffe_pb2


class TestCustomLayerMapping(unittest.TestCase):
    def test_extractor_custom_layer(self):
        expected_conv_params = {
            'num_output': 64,
            'pad': 1,
            'kernel_size': 3,
            'stride': 1,
            'bias_term': True,
            'axis': 1,
            'engine': 'caffe.ConvolutionParameter.DEFAULT',
            'group': 1,
            'force_nd_im2col': False,
            'pad_h': 0,
            'pad_w': 0
        }
        layer = """
                name: "conv"
                type: "Convolution"
                bottom: "input"
                top: "conv"
                convolution_param {
                    num_output: 64
                    pad: 1
                    kernel_size: 3
                    stride: 1
                }
                """
        mapping = {
            'NativeType': 'Convolution',
            'hasParam': 'true',
            'protoParamName': 'convolution_param'
        }
        proto = caffe_pb2.LayerParameter()
        text_format.Merge(layer, proto)
        attrs = proto_extractor(proto, None, mapping, False, False)
        for key, val in expected_conv_params.items():
            if key == 'bias_term' or key == 'force_nd_im2col':
                self.assertTrue(str(int(val)) == attrs[key])
            else:
                self.assertTrue(str(val) == attrs[key])
