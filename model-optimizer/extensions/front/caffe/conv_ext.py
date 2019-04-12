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

from mo.front.caffe.extractors.utils import get_spatial_attr, get_list_from_container, weights_biases
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.convolution import Convolution
from mo.utils.error import Error


class ConvFrontExtractor(FrontExtractorOp):
    op = 'convolution'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer, model_layer = node.pb, node.model_pb

        if not proto_layer:
            raise Error('Protobuf layer can not be empty')

        conv_param = proto_layer.convolution_param
        conv_type = 'ConvND' if len(proto_layer.bottom) > 1 else 'Conv2D'

        params = conv_set_params(conv_param, conv_type)
        attrs = conv_create_attrs(params)
        attrs.update({'op': conv_type,
                      'get_group': lambda node: node.group,
                      'get_output_feature_dim': lambda node: node.output
                      })

        # Embed weights and biases as attributes
        # It will be moved to a separate nodes in special pass
        attrs.update(
            weights_biases(conv_param.bias_term, model_layer, start_index=len(proto_layer.bottom), proto=conv_param))
        attrs.update(layout_attrs())

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return __class__.enabled


class DeconvFrontExtractor(FrontExtractorOp):
    op = 'deconvolution'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer, model_layer = node.pb, node.model_pb

        if not proto_layer:
            raise Error('Protobuf layer can not be empty')

        deconv_param = proto_layer.convolution_param

        params = conv_set_params(deconv_param, 'Deconv2D')
        attrs = conv_create_attrs(params)
        attrs.update({'type': 'Deconvolution',
                      'op': 'Deconv2D',
                      'get_group': lambda node: node.group,
                      'get_output_feature_dim': lambda node: node.output,
                      'input_feature_channel': 0,
                      'output_feature_channel': 1,
                      })

        # Embed weights and biases as attributes
        # It will be moved to a separate nodes in special pass
        attrs.update(weights_biases(deconv_param.bias_term, model_layer))
        attrs.update(layout_attrs())

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return __class__.enabled


def conv_create_attrs(params):
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
        'bias_addable': True,
        'bias_term': params['bias_term'],
        'pad': np.array([[0, 0], [0, 0], [params['padding'][1], params['padding'][1]],
                         [params['padding'][0], params['padding'][0]]], dtype=np.int64),
        'pad_spatial_shape': np.array([[params['padding'][1], params['padding'][1]],
                                       [params['padding'][0], params['padding'][0]]], dtype=np.int64),
        'dilation': np.array([1, 1, params['dilate'][1], params['dilate'][0]], dtype=np.int64),
        'output_spatial_shape': None,
        'output_shape': None,
        'stride': np.array([1, 1, params['stride'][1], params['stride'][0]], dtype=np.int64),
        'group': params['group'],
        'output': params['output'],
        'kernel_spatial': np.array([params['kernel'][1], params['kernel'][0]], dtype=np.int64),
        'kernel_spatial_idx': np.array([2, 3], dtype=np.int64),
        'reshape_kernel': True,

        'input_feature_channel': 1,
        'output_feature_channel': 0,
    }


def conv_set_params(conv_param, conv_type):
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
