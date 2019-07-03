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

from mo.front.common.partial_infer.utils import convert_tf_padding_to_str
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_data_format_spatial, tf_data_format_channel, tf_data_format_batch, \
    tf_int_list
from mo.ops.pooling import Pooling


class AvgPoolFrontExtractor(FrontExtractorOp):
    op = 'AvgPool'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = create_pooling_attrs(node, 'avg')
        attrs.update({'op': __class__.op})
        # update the attributes of the node
        Pooling.update_node_stat(node, attrs)
        return __class__.enabled


class MaxPoolFrontExtractor(FrontExtractorOp):
    op = 'MaxPool'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = create_pooling_attrs(node, 'max')
        attrs.update({'op': __class__.op})
        # update the attributes of the node
        Pooling.update_node_stat(node, attrs)
        return __class__.enabled


class MaxPool3DFrontExtractor(FrontExtractorOp):
    op = 'MaxPool3D'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = create_pooling_attrs(node, 'max')
        attrs.update({'op': __class__.op})
        # update the attributes of the node
        Pooling.update_node_stat(node, attrs)
        return __class__.enabled


class AvgPool3DFrontExtractor(FrontExtractorOp):
    op = 'AvgPool3D'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = create_pooling_attrs(node, 'avg')
        attrs.update({'op': __class__.op})
        # update the attributes of the node
        Pooling.update_node_stat(node, attrs)
        return __class__.enabled


def create_pooling_attrs(node, pool_method):
    data_format = node.pb.attr["data_format"]

    attrs = {
        'auto_pad': convert_tf_padding_to_str(node.pb.attr['padding']),
        'window': tf_int_list(node.pb.attr["ksize"].list),
        'spatial_dims': tf_data_format_spatial(data_format),
        'pad': None,  # will be inferred when input shape is known
        'stride': tf_int_list(node.pb.attr["strides"].list),
        'pad_spatial_shape': None,
        'output_spatial_shape': None,
        'pool_method': pool_method,
        'type': 'Pooling',
        'layout': data_format.s.decode(),
        'exclude_pad': 'true',
    }
    return attrs