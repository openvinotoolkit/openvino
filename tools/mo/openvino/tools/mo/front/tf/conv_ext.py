# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import convert_tf_padding_to_str, int64_array, dynamic_dimension
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_data_format_channel, tf_data_format_batch, \
    tf_int_list
from openvino.tools.mo.ops.convolution import Convolution
from openvino.tools.mo.ops.op import PermuteAttrs


class Conv2DFrontExtractor(FrontExtractorOp):
    op = 'Conv2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = tf_create_attrs(node, 2, 3)

        def get_num_groups(node):
            if 'group' in node:
                return node.group
            elif node.in_node(0).shape is not None and node.kernel_shape is not None \
                    and node.in_node(0).shape[node.channel_dims[0]] is not dynamic_dimension \
                    and node.kernel_shape[node.input_feature_channel] is not dynamic_dimension:
                # if group attribute is not defined, number of groups is calculated
                # from number of input channels and filter channel size
                return node.in_node(0).shape[node.channel_dims] // node.kernel_shape[node.input_feature_channel]
            else:
                return 1

        attrs.update({'op': __class__.op,
                      'get_group': get_num_groups,
                      'get_output_feature_dim': lambda node: node.kernel_shape[node.output_feature_channel],
                      'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([3, 2, 0, 1]),
                                                                      inv=int64_array([2, 3, 1, 0]))
                      })

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return cls.enabled


class DepthwiseConv2dNativeFrontExtractor(FrontExtractorOp):
    op = 'DepthwiseConv2dNative'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = tf_create_attrs(node, 2, 2)
        attrs.update({'op': __class__.op,
                      'kernel_spatial_idx': int64_array([0, 1]),
                      'get_group': lambda node: node.kernel_shape[node.output_feature_channel],
                      'get_output_feature_dim': lambda node: node.kernel_shape[-1] * node.kernel_shape[-2],
                      'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([2, 3, 0, 1]),
                                                                      inv=int64_array([2, 3, 0, 1]))
                      })

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return cls.enabled


class Conv3DFrontExtractor(FrontExtractorOp):
    op = 'Conv3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = tf_create_attrs(node, 3, 4)
        attrs.update({'op': __class__.op,
                      'get_group': lambda node: 1,
                      'get_output_feature_dim': lambda node: node.kernel_shape[node.output_feature_channel],
                      'get_weights_permute': PermuteAttrs.Permutation(perm=int64_array([4, 3, 0, 1, 2]),
                                                                      inv=int64_array([2, 3, 4, 1, 0]))
                      })

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return cls.enabled


def tf_create_attrs(node, input_feature_channel, output_feature_channel):
    data_format = node.pb.attr["data_format"]
    dilations = tf_int_list(node.pb.attr["dilations"].list)
    if len(dilations) == 0:
        dilations = None

    attrs = {
        'type': 'Convolution',
        'auto_pad': convert_tf_padding_to_str(node.pb.attr['padding'].s.decode()),
        'bias_addable': True,
        'bias_term': False,
        'dilation': dilations,
        'stride': tf_int_list(node.pb.attr["strides"].list),

        'channel_dims': tf_data_format_channel(data_format),
        'batch_dims': tf_data_format_batch(data_format),

        'input_feature_channel': input_feature_channel,
        'output_feature_channel': output_feature_channel,
        'layout': data_format.s.decode(),

        # get_group and get_output_feature_dim are special attrs that stores lambdas ( lambda node, kernel_shape:...)
        # this attrs calls in infer function to calculate output feature dimension and group attr
        'get_group': None,  # lambda should return group attr for given node
        'get_output_feature_dim': None,  # lamda should return output feature dimension
    }

    return attrs
