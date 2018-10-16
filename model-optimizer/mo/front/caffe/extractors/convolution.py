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

from mo.front.caffe.extractors.utils import weights_biases, get_spatial_attr, get_list_from_container
from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.convolution import caffe_conv2d_infer


def create_attrs(params):
    """
    Creates object of attrs for convolution
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
        object with all necessary convolution attributes

    """
    return {
        'type': 'Convolution',
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


def convolution_ext(proto_layer, model_layer):
    assert proto_layer, 'Protobuf layer can not be empty'
    conv_param = proto_layer.convolution_param

    conv_type = 'Conv2D'
    if len(proto_layer.bottom) > 1:
        conv_type = 'ConvND'

    params = set_params(conv_param, conv_type)
    attrs = create_attrs(params)
    # Embed weights and biases as attributes
    # It will be moved to a separate nodes in special pass
    attrs.update(
        weights_biases(conv_param.bias_term, model_layer, start_index=len(proto_layer.bottom), proto=conv_param))
    attrs.update(layout_attrs())
    return attrs


def set_params(conv_param, conv_type):
    # Defaults
    padding = [0, 0]
    stride = [1, 1]
    kernel = [0, 0]
    dilate = [1, 1]
    group = 1

    kernel = get_spatial_attr(kernel, 'kernel_size', 'kernel', conv_param)
    padding = get_spatial_attr(padding, 'pad', 'pad', conv_param)
    stride = get_spatial_attr(stride, 'stride', 'stride', conv_param)
    dilates = get_list_from_container(conv_param, 'dilation', int)
    if len(dilates) > 0:
        dilate[0] = dilate[1] = dilates[0]

    groups = get_list_from_container(conv_param, 'group', int)
    group = groups[0] if len(groups) > 0 and groups[0] != 1 else group

    return {
        'type_str': conv_type,
        'padding': padding,
        'dilate': dilate,
        'stride': stride,
        'kernel': kernel,
        'group': group,
        'output': conv_param.num_output,
        'bias_term': conv_param.bias_term
    }
