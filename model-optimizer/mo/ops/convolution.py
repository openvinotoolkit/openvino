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

import logging as log

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.utils import int64_array, float_array, mark_input_bins, assign_dims_to_weights, \
    tf_window_op_pad_infer
from mo.front.extractor import spatial_getter
from mo.front.onnx.extractors.utils import get_backend_pad
from mo.graph.graph import Node
from mo.ops.op import Op, PermuteAttrs
from mo.utils.error import Error


class Convolution(Op):
    op = 'Convolution'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
        }, attrs)

    def backend_attrs(self):
        return [
           'auto_pad',
           'group',
           ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
           ('dilations', lambda node: ','.join(map(str, node['dilation'][node.spatial_dims]))),
           ('kernel', lambda node: ','.join(map(str, node['kernel_spatial']))),

           ('pads_begin', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 0)))),
           ('pads_end', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 1)))),
           'output'
        ]

    def backend_attrs_v2(self):
        return [
            spatial_getter('stride-x', 'stride', 1),
            spatial_getter('stride-y', 'stride', 0),

            ('kernel-x', lambda node: node.kernel_spatial[1]),
            ('kernel-y', lambda node: node.kernel_spatial[0]),

            spatial_getter('dilation-x', 'dilation', 0),
            spatial_getter('dilation-y', 'dilation', 1),
            spatial_getter('pad-x', 'pad', 1, lambda x: x[0]),
            spatial_getter('pad-y', 'pad', 0, lambda x: x[0]),
            spatial_getter('pad-r', 'pad', 1, lambda x: x[1]),
            spatial_getter('pad-b', 'pad', 0, lambda x: x[1]),

            'auto_pad',
            'output',
            'group',
        ]


    @staticmethod
    def calc_convolution(input_spatial_shape, stride_spatial_shape, pad_spatial_shape, kernel_extent):
        ''' Calculates output shape for Convolution.
            Verified to be applicable for both Caffe and ONNX.
        '''
        spatial_val_wo_stride = input_spatial_shape + pad_spatial_shape - kernel_extent
        float_spatial_val_wo_stride = float_array(spatial_val_wo_stride)
        return float_spatial_val_wo_stride / stride_spatial_shape + 1

    @staticmethod
    def calc_deconvolution(node, input_spatial_shape, pad_spatial_shape, kernel_extent, output_padding=None):
        ''' Calculates output shape for Deconvolution.
            Verified to be applicable for both Caffe and ONNX with explicitly defined pads.
            If pads are not specified for ONNX operator, this function is not applicable.
        '''
        shape = node.stride[node.spatial_dims] * (input_spatial_shape - 1) + kernel_extent - pad_spatial_shape
        if output_padding is not None:
            shape += output_padding
        return shape

    @staticmethod
    def infer(node: Node):
        """
        Infers shape of convolution node as it is done in ONNX.
        It is very similar to one that Caffe does, but slightly different.
        We made a complete fork of this function because they are supposed to be
        supported differently by different people.
        Args:
            node: graph convolution node
        """
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        # bias_term cannot be deduced earlier for frameworks that represent
        # convolution weights/biases as regular inputs; so the number of inputs
        # is being checked here and restore correct value for bias_term to
        # have the rest of the code unchanged. It will be used after we merge
        # several infer functions for convolution in different FWs to a single one.
        if not node.has_valid('bias_term'):
            node['bias_term'] = len(node.in_nodes()) == 3

        # In case of caffe we have to calculate input index for weights because
        # caffe convolution can be with more than one input
        weights_index = len(node.in_nodes()) - 2
        if not node.bias_term:
            weights_index = len(node.in_nodes()) - 1

        # Reshape weights kernel to original shape
        # In case of caffe ot MXNet framework, values for weights has no structed shape like OIHW
        # so we have to reshape weights to normal shape
        # For this case, Convolution node should have attribute reshape_kernel = True
        if node.has_valid('reshape_kernel') and node.reshape_kernel:
            if not (node.has_valid('output') and node.has_valid('channel_dims') and node.has_valid(
                    'group') and node.has_valid('kernel_spatial')):
                log.error('Can\'t reshape kernel due to not all required attrs was set to {} node'.format(node.id))
                return
            # layout for Convolution weights is OIHW
            kernel_shape = np.array([node.output, input_shape[node.channel_dims].item() / node.group,
                                    *[node.kernel_spatial[i] for i in range(len(node.kernel_spatial))]], dtype=np.int64)
            if node.type == 'Deconvolution':  # layout for Deconvolution weights is IOHW
                kernel_shape[[0, 1]] = kernel_shape[[1, 0]]

            node.in_node(weights_index).shape = np.array(kernel_shape)
            node.in_node(weights_index).value = np.reshape(node.in_node(weights_index).value, kernel_shape)
            node.reshape_kernel = False

        # Pass weights shape to node attribute kernel_shape
        kernel_shape = node.in_node(weights_index).shape
        node['kernel_shape'] = kernel_shape
        # Calculate kernel_spatial_idx and spatial_dims if it is not specified
        # It is necessary for ONNX dut to convolution can be 1D/2D/3D
        if not node.has_valid('kernel_spatial_idx'):
            node['kernel_spatial_idx'] = np.delete([x for x in range(len(kernel_shape))], (node.input_feature_channel, node.output_feature_channel))

        if not node.has_valid('spatial_dims'):
            node['spatial_dims'] = np.delete([x for x in range(len(input_shape))], (node.channel_dims[0], node.batch_dims[0]))

        node['kernel_spatial'] = kernel_shape[node.kernel_spatial_idx]

        if not node.has_valid('output'):
            # restore the number of output feature maps from the scond argument that is weights
            if node.type in ['Convolution', 'Deconvolution']:
                node['output'] = kernel_shape[node.output_feature_channel]
            else:
                raise Error(
                    'Convolution infer function was called for a node {} with unsupported type {}',
                    node.soft_get('name'),
                    node.type
                )

        # Set default values for dilation, strides and pads if not set
        if not node.has_valid('dilation'):
            node['dilation'] = np.full([len(input_shape)], 1, dtype=np.int64)
        if not node.has_valid('stride'):
            node['stride'] = np.full([len(input_shape)], 1, dtype=np.int64)
        if not node.has_valid('pad'):
            node['pad'] = np.array([[0, 0]] * len(input_shape), dtype=np.int64)
            node['pad_spatial_shape'] = node.pad[node.spatial_dims]

        input_spatial_shape = input_shape[node.spatial_dims]
        stride_spatial_shape = node.stride[node.spatial_dims]

        kernel_extent = node.dilation[node.spatial_dims] * (node.kernel_spatial - 1) + 1
        # TensorFlow always has auto_pad attribute that can be either valid or same_upper
        # In ONNX auto_pad attribute is deprecated but appears in some models (could be valid, same_upper or same_lower)
        # Caffe do not use auto_pad attribute
        if node.has_valid('auto_pad') and not node.has_valid('output_spatial_shape'):
            node['pad_spatial_shape'], node['output_spatial_shape'] = tf_window_op_pad_infer(input_spatial_shape,
                                                                                       kernel_extent,
                                                                                       stride_spatial_shape,
                                                                                       node.auto_pad)
            pad = np.zeros((len(input_shape), 2), dtype=np.int64)
            pad[node.spatial_dims] = node.pad_spatial_shape
            node.pad = pad
        else:
            pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)
            if node.type == 'Convolution':
                float_spatial = Convolution.calc_convolution(input_spatial_shape, stride_spatial_shape,
                                                             pad_spatial_shape,
                                                             kernel_extent)
                node['output_spatial_shape'] = int64_array(float_spatial)
            elif node.type == 'Deconvolution':
                # In case of given output_spatial_shape we calculate pads spatial
                if node.has_valid('output_spatial_shape'):
                    if node.has_valid('get_pad'):
                        node['pad'] = node.get_pad(node, input_shape, kernel_shape)
                    else:
                        log.debug('Can\'t calculate paddings due to missing lambda get_pad in {} node'.format(node.id))
                        return
                else:
                    output_padding = node.output_padding[node.spatial_dims] if node.has_valid('output_padding') else None
                    float_spatial = Convolution.calc_deconvolution(node, input_spatial_shape, pad_spatial_shape,
                                                                   kernel_extent,
                                                                   output_padding)
                    node['output_spatial_shape'] = int64_array(float_spatial)
            else:
                return

        # For cases when group attribute wasn't set in extractor we should specify get_group attribute
        # this attribute should store lambda node: ... (check tf convolution extractor)
        if node.has_valid('get_group'):
            node['group'] = node.get_group(node)
        output_shape = np.full_like(input_shape, -1, dtype=np.int64)
        output_shape[node.batch_dims] = input_shape[node.batch_dims]
        output_shape[node.spatial_dims] = node.output_spatial_shape

        # For cases when output attribute wasn't set in extractor we should specify get_output_feature_dim attribute
        # this attribute should store lambda node: ... (check tf convolution extractor)
        if node.has_valid('get_output_feature_dim'):
            node['output'] = node.get_output_feature_dim(node)
        output_shape[node.channel_dims] = node.output
        node['output_shape'] = output_shape

        for n in node.out_nodes():
            node.out_node(n).shape = output_shape

        mark_input_bins(node)
        assign_dims_to_weights(node.in_node(weights_index), node.kernel_spatial_idx, node.input_feature_channel,
                               node.output_feature_channel, len(kernel_shape))

        PermuteAttrs.create_permute_attrs(node, attrs=[('pad', 'input:0'),
                                                       ('stride', 'input:0'),
                                                       ('dilation', 'input:0'),
                                                       ('output_shape', 'input:0'),
                                                       ('batch_dims', 'input:0'),
                                                       ('channel_dims', 'input:0'),
                                                       ('spatial_dims', 'input:0'),

                                                       ('kernel_shape', 'input:{}'.format(weights_index)),
                                                       ('kernel_spatial_idx', 'input:{}'.format(weights_index)),
                                                       ('input_feature_channel', 'input:{}'.format(weights_index)),
                                                       ('output_feature_channel', 'input:{}'.format(weights_index)),
                                                       ])

        PermuteAttrs.set_permutation(node.in_node(weights_index), node,
                                     node.get_weights_permute if node.has_valid('get_weights_permute') else None,
                                     skip_if_exists=True)
