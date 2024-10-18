# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, mark_input_bins, assign_dims_to_weights, \
    tf_window_op_pad_infer, dynamic_dimension_value, shape_array, is_fully_defined, undefined_shape_of_rank
from openvino.tools.mo.front.onnx.extractors.utils import get_backend_pad
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op, PermuteAttrs
from openvino.tools.mo.pipeline.common import convert_const_node_value_type
from openvino.tools.mo.utils.error import Error


class Convolution(Op):
    op = 'Convolution'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'type_infer': self.type_infer,
            'multiplication_transparent': True,
            'multiplication_transparent_ports': [(0, 0), (1, 0)],
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        def pad_attribute_helper(node: Node, pad_type: str='begin'):
            assert pad_type in ['begin', 'end']
            if not node.has_valid('pad'):
                return None
            pad = get_backend_pad(node.pad, node.spatial_dims, 0 if pad_type == 'begin' else 1)
            if node.has_valid('auto_pad') and node.auto_pad != 'explicit':
                pad = [0 for _ in pad]
            return ','.join(map(str, pad))

        return [
            ('auto_pad', lambda node: node.auto_pad if node.has_valid('auto_pad') else 'explicit'),
            ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
            ('dilations', lambda node: ','.join(map(str, node['dilation'][node.spatial_dims]))),
            ('pads_begin', lambda node: pad_attribute_helper(node, 'begin')),
            ('pads_end', lambda node: pad_attribute_helper(node, 'end')),

            # for Backpropdata operations only - according to spec
            ('output_padding', lambda node: ','.join(map(str, node.output_padding[node.spatial_dims])) \
                if node.has_valid('output_padding') and node.type in
                    ('GroupConvolutionBackpropData', 'ConvolutionBackpropData') else None),

            # for BinaryConvolution only
            'pad_value',
            'mode',
        ]

    @staticmethod
    def calc_convolution(input_spatial_shape, stride_spatial_shape, pad_spatial_shape, kernel_extent):
        """
        Calculates output shape for Convolution.
        Verified to be applicable for both Caffe and ONNX.
        """
        spatial_val_wo_stride = input_spatial_shape + pad_spatial_shape - kernel_extent

        if np.any(spatial_val_wo_stride < 0):
            raise Error("Data after padding has dimension less than window size. " +
                        "Possible reason of error is incorrectly specified model input shape(s).")

        return spatial_val_wo_stride / stride_spatial_shape + 1

    @staticmethod
    def calc_deconvolution(node, input_spatial_shape, pad_spatial_shape, kernel_extent):
        """
        Calculates output shape for Deconvolution.
        Verified to be applicable for both Caffe and ONNX with explicitly defined pads.
        If pads are not specified for ONNX operator, this function is not applicable.
        """
        return node.stride[node.spatial_dims] * (input_spatial_shape - 1) + kernel_extent - pad_spatial_shape

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
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            raise Error('Input data shape is None for node {}'.format(node.soft_get('name', node.id)))

        # bias_term cannot be deduced earlier for frameworks that represent
        # convolution weights/biases as regular inputs; so the number of inputs
        # is being checked here and restore correct value for bias_term to
        # have the rest of the code unchanged. It will be used after we merge
        # several infer functions for convolution in different FWs to a single one.
        if not node.has_valid('bias_term'):
            node['bias_term'] = len(node.in_nodes()) == 3

        weights_index = node.weights_index if node.has_valid('weights_index') else 1
        # Reshape weights kernel to original shape
        # In case of Caffe framework, values for weights have no structured shape like OIHW
        # so we have to reshape weights to normal shape
        # For this case, Convolution node should have attribute reshape_kernel = True
        if node.has_valid('reshape_kernel') and node.reshape_kernel:
            if not (node.has_valid('output') and node.has_valid('channel_dims') and node.has_valid(
                    'group') and node.has_valid('kernel_spatial')):
                log.error('Cannot reshape kernel due to not all required attrs was set to {} node'.format(node.id))
                return

            # since item() unmasks values, result should be masked back
            num_in_channels = shape_array(input_shape[node.channel_dims].item())

            # layout for Convolution weights is OIHW
            kernel_shape = shape_array([node.output, num_in_channels / node.group,
                                       *[node.kernel_spatial[i] for i in range(len(node.kernel_spatial))]])
            if node.type == 'Deconvolution':  # layout for Deconvolution weights is IOHW
                kernel_shape[[0, 1]] = kernel_shape[[1, 0]]

            if is_fully_defined(kernel_shape) and np.prod(kernel_shape) != np.prod(node.in_node(weights_index).value.shape):
                log.error("Size of weights {} does not match kernel shape: {}\n"
                          "".format(np.prod(node.in_node(weights_index).value.shape), kernel_shape) +
                          "    Possible reason is wrong channel number in input shape\n")
                raise Error("Cannot reshape weights to kernel shape")

            if not is_fully_defined(kernel_shape):
                num_undefined = np.count_nonzero(kernel_shape.mask is True)  # pylint: disable=no-member
                if num_undefined > 1:
                    raise Error('Too many undefined dimensions of the kernel shape for node {}. Use --input_shape '
                                'command line parameter to specify model input shapes'.format(node.soft_get('name',
                                                                                                            node.id)))
                kernel_size = np.prod(node.in_node(weights_index).value.shape)
                # calculate undefined dimension using fully defined shape of the weights input and known kernel_shape
                # dimensions
                kernel_shape[np.where(kernel_shape == np.ma.masked)[0][0]] = kernel_size // np.prod(kernel_shape)

            node.in_node(weights_index).shape = shape_array(kernel_shape)
            node.in_node(weights_index).value = np.reshape(node.in_node(weights_index).value, kernel_shape)
            node.reshape_kernel = False

        # Pass weights shape to node attribute kernel_shape
        kernel_shape = node.in_node(weights_index).shape
        node['kernel_shape'] = kernel_shape
        # Calculate kernel_spatial_idx and spatial_dims if it is not specified
        # It is necessary for ONNX dut to convolution can be 1D/2D/3D
        if not node.has_valid('kernel_spatial_idx'):
            node['kernel_spatial_idx'] = np.delete([x for x in range(len(kernel_shape))],
                                                   (node.input_feature_channel, node.output_feature_channel))

        if not node.has_valid('spatial_dims'):
            node['spatial_dims'] = np.delete([x for x in range(len(input_shape))],
                                             (node.channel_dims[0], node.batch_dims[0]))

        node['kernel_spatial'] = kernel_shape[node.kernel_spatial_idx]

        if not node.has_valid('output'):
            # restore the number of output feature maps from the second argument that is weights
            if node.type in ['Convolution', 'Deconvolution', 'DeformableConvolution', 'BinaryConvolution']:
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
            node['pad'] = int64_array([[0, 0]] * len(input_shape))
        node['pad_spatial_shape'] = node.pad[node.spatial_dims]

        if not node.has_valid('output_padding'):
            node['output_padding'] = np.full([len(input_shape)], 0, dtype=np.int64)

        if node.has_valid('output_padding') and len(input_shape) > len(node['output_padding']):
            output_padding = np.zeros(len(input_shape), dtype=np.int64)
            for i in range(len(node['output_padding'])):
                output_padding[i] = node['output_padding'][i]
            node['output_padding'] = output_padding

        input_spatial_shape = input_shape[node.spatial_dims]
        stride_spatial_shape = node.stride[node.spatial_dims]

        kernel_extent = node.dilation[node.spatial_dims] * (node.kernel_spatial - 1) + 1
        # TensorFlow always has auto_pad attribute that can be either valid or same_upper
        # In ONNX auto_pad attribute is deprecated but appears in some models (could be valid, same_upper or same_lower)
        # Caffe do not use auto_pad attribute
        if node.has_valid('auto_pad') and node.auto_pad != 'explicit' and not node.has_valid('output_spatial_shape'):
            node['pad_spatial_shape'], node['output_spatial_shape'] = tf_window_op_pad_infer(input_spatial_shape,
                                                                                             kernel_extent,
                                                                                             stride_spatial_shape,
                                                                                             node.auto_pad,
                                                                                             node.type == 'Deconvolution')

            pad = np.zeros((len(input_shape), 2), dtype=np.int64)
            pad[node.spatial_dims] = node.pad_spatial_shape
            node.pad = pad
        else:
            pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)
            if node.type in ('Convolution', 'BinaryConvolution'):
                float_spatial = Convolution.calc_convolution(input_spatial_shape, stride_spatial_shape,
                                                             pad_spatial_shape,
                                                             kernel_extent)
                node['output_spatial_shape'] = shape_array(float_spatial)
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
                    if output_padding is not None and any(output_padding):
                        pad_spatial_shape -= output_padding
                        for dim in range(len(pad_spatial_shape)):
                            node.pad_spatial_shape[dim][1] -= pad_spatial_shape[dim]

                    float_spatial = Convolution.calc_deconvolution(node, input_spatial_shape, pad_spatial_shape,
                                                                   kernel_extent)
                    node['output_spatial_shape'] = shape_array(float_spatial)
            elif node.type == 'DeformableConvolution':
                # get the output spatial shape from the second input with offsets
                node['output_spatial_shape'] = int64_array([node.in_node(1).shape[2:4]])
            else:
                assert 'Unsupported layer type "{}"'.format(node.type)

        # For cases when group attribute wasn't set in extractor we should specify get_group attribute
        # this attribute should store lambda node: ... (check tf convolution extractor)
        if node.has_valid('get_group'):
            node['group'] = node.get_group(node)
        output_shape = shape_array([dynamic_dimension_value for _ in range(len(input_shape))])
        output_shape[node.batch_dims] = input_shape[node.batch_dims]  # pylint: disable=unsupported-assignment-operation
        output_shape[node.spatial_dims] = node.output_spatial_shape  # pylint: disable=unsupported-assignment-operation

        # For cases when output attribute wasn't set in extractor we should specify get_output_feature_dim attribute
        # this attribute should store lambda node: ... (check tf convolution extractor)
        if node.has_valid('get_output_feature_dim'):
            node['output'] = node.get_output_feature_dim(node)
        output_shape[node.channel_dims] = node.output  # pylint: disable=unsupported-assignment-operation
        node['output_shape'] = output_shape

        node.out_port(0).data.set_shape(output_shape)

        # bin attribute is used for pre-processing, but it will be deleted in BlobNormalizer transformation
        # and the blobs (weights, biases) will be represented as inputs to the node
        mark_input_bins(node, start_port=1 if node.type != 'DeformableConvolution' else 2)
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

        # is needed to permute Conv weights from the original TF [H, W, C_IN, C_OUT] into OV [C_OUT, C_IN, H, W]
        # but for other nodes in weights subgraph permutations must turned off
        # by marking with MarkSubGraphsWithCorrectLayout even if graph layout is NCHW.
        PermuteAttrs.set_permutation(node.in_node(weights_index), node, node.soft_get('get_weights_permute', None))
        PermuteInputs().set_input_permutation(
            node.in_node(weights_index), node, 'input:{}'.format(weights_index), 'transpose')

    @staticmethod
    def reverse_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            shape = None
            # TODO FIXME this is ugly solution based on various attributes which may not be set in some cases
            for attr in ['dilation', 'stride', 'pad']:
                if node.has_valid(attr):
                    shape = undefined_shape_of_rank(len(node.soft_get(attr)))
                    break
            if shape is not None:
                node.in_port(0).data.set_shape(shape)

    @staticmethod
    def type_infer(node):
        in_type_0 = node.in_port(0).get_data_type()
        in_type_1 = node.in_port(1).get_data_type()
        in_node_1 = node.in_port(1).get_source().node
        # in case of input values data type mismatch we try to change the type of the constant to match the type of
        # input at index 0.
        if in_type_1 in [np.float16, np.float32, np.float64] and in_type_0 != in_type_1 and in_node_1.op == 'Const':
            in_node_1 = node.in_port(1).get_source().node
            log.error("Changing Const node '{}' data type from {} to {} for Convolution operation".format(
                in_node_1.soft_get('name', in_node_1.id), in_type_1, in_type_0),
                extra={'is_warning': True})
            convert_const_node_value_type(in_node_1, in_type_0)
        node.out_port(0).set_data_type(node.in_port(0).get_data_type())
