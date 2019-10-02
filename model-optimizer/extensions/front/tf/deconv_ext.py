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

from mo.front.common.partial_infer.utils import convert_tf_padding_to_str, int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_data_format_spatial, tf_data_format_channel, tf_data_format_batch, \
    tf_int_list
from mo.ops.deconvolution import Deconvolution
from mo.ops.op import PermuteAttrs


class Conv2DBackpropInputFrontExtractor(FrontExtractorOp):
    op = 'Conv2DBackpropInput'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = tf_create_attrs(node, 3, 2)
        attrs.update({'op': __class__.op,
                      'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([3, 2, 0, 1]),
                                                                      inv=int64_array([2, 3, 1, 0]))
                      })

        # update the attributes of the node
        Deconvolution.update_node_stat(node, attrs)
        return __class__.enabled


class Conv3DBackpropInputV2InputFrontExtractor(FrontExtractorOp):
    op = 'Conv3DBackpropInputV2'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = tf_create_attrs(node, 4, 3)
        attrs.update({'op': __class__.op,
                      'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([4, 3, 0, 1, 2]),
                                                                      inv=int64_array([2, 3, 4, 1, 0]))
                      })

        # update the attributes of the node
        Deconvolution.update_node_stat(node, attrs)
        return __class__.enabled


def tf_create_attrs(node, input_feature_channel, output_feature_channel):
    data_format = node.pb.attr["data_format"]

    return {
        'auto_pad': convert_tf_padding_to_str(node.pb.attr['padding']),
        'bias_addable': True,
        'bias_term': False,
        'spatial_dims': tf_data_format_spatial(data_format),
        'channel_dims': tf_data_format_channel(data_format),
        'batch_dims': tf_data_format_batch(data_format),
        'pad': None,  # will be inferred when input shape is known
        'pad_spatial_shape': None,
        'output_spatial_shape': None,
        'output_shape': None,
        'output': None,
        'stride': tf_int_list(node.pb.attr["strides"].list),
        'type': None,  # don't set type until we are sure it is really translated to correct IR; see infer function
        'group': None,
        'layout': data_format.s.decode(),
        'input_feature_channel': input_feature_channel,
        'output_feature_channel': output_feature_channel,
    }
