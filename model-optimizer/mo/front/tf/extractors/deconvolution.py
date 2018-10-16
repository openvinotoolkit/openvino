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

from mo.front.common.partial_infer.deconvolution import tf_deconv2d_infer
from mo.front.tf.extractors.utils import tf_data_format_spatial, tf_data_format_channel, tf_data_format_batch, \
    tf_int_list
from mo.front.common.partial_infer.utils import convert_tf_padding_to_str


def tf_deconv2d_ext(pb):
    data_format = pb.attr["data_format"]

    return {
        'auto_pad': convert_tf_padding_to_str(pb.attr['padding']),
        'bias_addable': True,
        'bias_term': False,
        'spatial_dims': tf_data_format_spatial(data_format),
        'channel_dims': tf_data_format_channel(data_format),
        'batch_dims': tf_data_format_batch(data_format),
        'pad': None,  # will be inferred when input shape is known
        'pad_spatial_shape': None,
        'output_spatial_shape': None,
        'output_shape': None,
        'stride': tf_int_list(pb.attr["strides"].list),
        'type': None,  # don't set type until we are sure it is really translated to correct IR; see infer function
        'group': None,
        'layout': 'NHWC',
        'infer': tf_deconv2d_infer
    }
