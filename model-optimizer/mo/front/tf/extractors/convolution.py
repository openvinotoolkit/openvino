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
import logging as log

from mo.utils.error import Error
from mo.front.common.partial_infer.convolution import tf_conv2d_infer
from mo.front.tf.extractors.utils import tf_data_format_spatial, tf_data_format_channel, tf_data_format_batch, \
    tf_int_list
from mo.front.common.partial_infer.utils import convert_tf_padding_to_str

def tf_conv2d_ext(pb, is_depthwise_conv: bool = False):
    data_format = pb.attr["data_format"]
    input_feature_channel, output_feature_channel = ([2], [2]) if is_depthwise_conv else ([2], [3])

    # Currently supported only [1,1,1,1] dilation
    dilations = tf_int_list(pb.attr["dilations"].list)
    if len(dilations) > 0 and not np.array_equal(np.array(dilations), np.array([1,1,1,1])):
        raise Error('Dilation {} for {} is not supported'.format(dilations, pb.name))

    return {
        'type': 'Convolution',

        'auto_pad': convert_tf_padding_to_str(pb.attr['padding']),
        'bias_addable': True,
        'bias_term': False,
        'pad': None,  # will be inferred when input shape is known
        'pad_spatial_shape': None,
        'dilation': np.array([1, 1, 1, 1], dtype=np.int64),
        'output_spatial_shape': None,
        'output_shape': None,
        'stride': tf_int_list(pb.attr["strides"].list),
        'group': None,

        'spatial_dims': tf_data_format_spatial(data_format),
        'channel_dims': tf_data_format_channel(data_format),
        'batch_dims': tf_data_format_batch(data_format),
        'kernel_spatial': None,
        'input_feature_channel': input_feature_channel,
        'output_feature_channel': output_feature_channel,
        'layout': 'NHWC',
        'infer': lambda node: tf_conv2d_infer(node, is_depthwise_conv)
    }
