# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from google.protobuf import text_format

from openvino.tools.mo.front.caffe.custom_layers_mapping import proto_extractor
from openvino.tools.mo.front.caffe.proto import caffe_pb2


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
