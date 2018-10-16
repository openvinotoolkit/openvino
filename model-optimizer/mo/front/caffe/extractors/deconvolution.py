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

from mo.front.caffe.extractors.convolution import set_params
from mo.front.caffe.extractors.utils import weights_biases
from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.convolution import caffe_conv2d_infer


def create_attrs(params):
    """
    Creates object of attrs for deconvolution
    Args:
        params: {
            type_str: type_str
            padding: padding
            dilate: dilate
            stride: stride
            kernel: kernel
            group: group
            output: output
            bias_term: bias_term
        }
    Returns:
        object with all necessary deconvolution attributes

    """
    return {
        'type': 'Deconvolution',
        'op': params['type_str'],
        'bias_addable': True,
        'bias_term': params['bias_term'],
        'pad': np.array([[0, 0], [0, 0],
                         [params['padding'][1], params['padding'][1]],
                         [params['padding'][0], params['padding'][0]]], dtype=np.int64),
        'pad_spatial_shape': np.array([[params['padding'][1], params['padding'][1]],
                                       [params['padding'][0], params['padding'][0]]], dtype=np.int64),
        'dilation': np.array([1, 1,
                              params['dilate'][1], params['dilate'][0]], dtype=np.int64),
        'output_spatial_shape': None,
        'output_shape': None,
        'stride': np.array([1, 1, params['stride'][1],
                            params['stride'][0]], dtype=np.int64),
        'infer': caffe_conv2d_infer,
        'group': params['group'],
        'output': params['output'],
        'kernel_spatial': np.array([params['kernel'][1], params['kernel'][0]], dtype=np.int64)
    }


def deconvolution_ext(proto_layer, model_layer):
    assert proto_layer, 'Protobuf layer can not be empty'
    deconv_param = proto_layer.convolution_param
    params = set_params(deconv_param, 'Deconv2D')
    attrs = create_attrs(params)
    # Embed weights and biases as attributes
    # It will be moved to a separate nodes in special pass
    attrs.update(weights_biases(deconv_param.bias_term, model_layer))
    attrs.update(layout_attrs())
    return attrs
