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

from mo.front.common.partial_infer.pooling import tf_pool_infer
from mo.front.tf.extractors.utils import tf_int_list, tf_data_format_spatial
from mo.front.common.partial_infer.utils import convert_tf_padding_to_str


def tf_pool_ext(pb, op=None, pool_method=None):
    data_format = pb.attr["data_format"]

    return {
        'auto_pad': convert_tf_padding_to_str(pb.attr['padding']),
        'window': tf_int_list(pb.attr["ksize"].list),
        'spatial_dims': tf_data_format_spatial(data_format),
        'pad': None,  # will be inferred when input shape is known
        'stride': tf_int_list(pb.attr["strides"].list),
        'pad_spatial_shape': None,
        'pool_method': pool_method,
        'type': 'Pooling',
        'exclude_pad': 'true',
        'rounding_type': 'floor',
        'infer': lambda node: tf_pool_infer(node, op)
    }
