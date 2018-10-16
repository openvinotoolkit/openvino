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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array, float_array, mark_input_bins, assign_dims_to_weights, \
    tf_window_op_pad_infer
from mo.utils.error import Error


def calc_convolution_caffe(input_spatial_shape, stride_spatial_shape, pad_spatial_shape, kernel_extent):
    ''' Calculates output shape for Convolution.
        Verified to be applicable for both Caffe and ONNX.
    '''
    spatial_val_wo_stride = input_spatial_shape + pad_spatial_shape - kernel_extent
    float_spatial_val_wo_stride = float_array(spatial_val_wo_stride)
    return float_spatial_val_wo_stride / stride_spatial_shape + 1


def calc_deconvolution_caffe(node, input_spatial_shape, pad_spatial_shape, kernel_extent, output_padding=None):
    ''' Calculates output shape for Deconvolution.
        Verified to be applicable for both Caffe and ONNX with explicitly defined pads.
        If pads are not specified for ONNX operator, this function is not applicable.
    '''
    shape = node.stride[node.spatial_dims] * (input_spatial_shape - 1) + kernel_extent - pad_spatial_shape
    if output_padding is not None:
        shape += output_padding
    return shape


def onnx_conv2d_infer(node):
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
    if node.bias_term is None:
        node.bias_term = len(node.in_nodes()) == 3

    input_spatial_shape = input_shape[node.spatial_dims]
    pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)
    stride_spatial_shape = node.stride[node.spatial_dims]

    if not node.has_valid('output'):
        # restore the number of output feature maps from the scond argument that is weights
        if node.type == 'Convolution':
            node['output'] = node.in_node(1).shape[0]
        elif node.type == 'Deconvolution':
            node['output'] = node.in_node(1).shape[1]
        else:
            raise Error(
                'onnx_conv2d_infer function was called for a node {} with unsupported type {}',
                node.soft_get('name'),
                node.type
            )

    output_channels = node.output

    kernel_extent = node.dilation[node.spatial_dims] * (node.kernel_spatial - 1) + 1

    if node.has_valid('auto_pad'):
        node.pad_spatial_shape, node.output_spatial_shape = tf_window_op_pad_infer(input_spatial_shape, kernel_extent,
                                                                                   stride_spatial_shape, node.auto_pad)
        pad = np.zeros((len(input_shape), 2), dtype=np.int64)
        pad[node.spatial_dims] = node.pad_spatial_shape
        node.pad = pad
    else:
        if node.type == 'Convolution':
            float_spatial = calc_convolution_caffe(input_spatial_shape, stride_spatial_shape, pad_spatial_shape, kernel_extent)
        elif node.type == 'Deconvolution':
            float_spatial = calc_deconvolution_caffe(node, input_spatial_shape, pad_spatial_shape, kernel_extent, node.output_padding)
        else:
            return
        node.output_spatial_shape = int64_array(float_spatial)

    output_shape = np.full_like(input_shape, -1, dtype=np.int64)
    output_shape[node.batch_dims] = input_shape[node.batch_dims]
    output_shape[node.channel_dims] = output_channels
    output_shape[node.spatial_dims] = node.output_spatial_shape

    node.output_shape = output_shape

    for n in node.out_nodes():
        node.out_node(n).shape = output_shape

    mark_input_bins(node)
    assign_dims_to_weights(node.in_node(1), [2, 3], [1], [0], 4)
    return


def caffe_conv2d_infer(node):
    """
    Infers shape of convolution node as it is done in Caffe.
    It also modifies shape of input weights data node.
    Link to the original sources:
    https://github.com/BVLC/caffe/blob/99466224dac86ddb86296b1e727794fb836bd80f/src/caffe/layers/conv_layer.cpp#L8
    Args:
        node: graph convolution node

    """
    input_shape = node.in_node(0).shape
    if input_shape is None:
        return

    input_spatial_shape = input_shape[node.spatial_dims]
    input_channels = input_shape[node.channel_dims]
    stride_spatial_shape = node.stride[node.spatial_dims]
    pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)
    output_channels = node.output

    kernel_extent = node.dilation[node.spatial_dims] * (node.kernel_spatial - 1) + 1
    if node.type == 'Convolution':
        float_spatial = calc_convolution_caffe(input_spatial_shape, stride_spatial_shape, pad_spatial_shape, kernel_extent)
    elif node.type == 'Deconvolution':
        float_spatial = calc_deconvolution_caffe(node, input_spatial_shape, pad_spatial_shape, kernel_extent)
    else:
        return

    node.output_spatial_shape = int64_array(float_spatial)

    output_shape = np.full_like(input_shape, -1, dtype=np.int64)
    output_shape[node.batch_dims] = input_shape[node.batch_dims]
    output_shape[node.channel_dims] = output_channels
    output_shape[node.spatial_dims] = node.output_spatial_shape

    node.output_shape = output_shape

    for n in node.out_nodes():
        node.out_node(n).shape = output_shape

    # Calculate kernel shape; it should be in OIYX layout
    num_inputs = len(node.in_nodes())
    if node.bias_term == False:
        weights_index = num_inputs - 1
    else:
        weights_index = num_inputs - 2

    if weights_index <= 0:
        log.error("No weights in convolution layer!" + node.id)
        return

    weights = node.in_node(weights_index)
    shape = [output_channels, input_channels / node.group, node.kernel_spatial[0], node.kernel_spatial[1]]
    weights.shape = int64_array(shape)
    weights.value.shape = weights.shape

    mark_input_bins(node, start_port=weights_index)
    assign_dims_to_weights(weights, [2, 3], [1], [0], 4)


def mxnet_conv2d_infer(node):
    """
    Infers shape of convolution node as it is done in Caffe.
    It also modifies shape of input weights data node.
    Link to the original sources:
    https://github.com/BVLC/caffe/blob/99466224dac86ddb86296b1e727794fb836bd80f/src/caffe/layers/conv_layer.cpp#L8
    Args:
        node: graph convolution node

    """
    input_shape = node.in_node(0).shape
    if input_shape is None:
        return

    input_spatial_shape = input_shape[node.spatial_dims]
    input_channels = input_shape[node.channel_dims]
    stride_spatial_shape = node.stride[node.spatial_dims]
    pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)
    output_channels = node.output

    kernel_extent = node.dilation[node.spatial_dims] * (node.kernel_spatial - 1) + 1
    if node.type == 'Convolution':
        float_spatial = calc_convolution_caffe(input_spatial_shape, stride_spatial_shape, pad_spatial_shape, kernel_extent)
    elif node.type == 'Deconvolution':
        float_spatial = calc_deconvolution_caffe(node, input_spatial_shape, pad_spatial_shape, kernel_extent)
    else:
        return

    node.output_spatial_shape = int64_array(float_spatial)

    output_shape = np.full_like(input_shape, -1, dtype=np.int64)
    output_shape[node.batch_dims] = input_shape[node.batch_dims]
    output_shape[node.channel_dims] = output_channels
    output_shape[node.spatial_dims] = node.output_spatial_shape

    node.output_shape = output_shape
    node.out_node().shape = output_shape

    weights = node.in_node(1)
    shape = [output_channels, input_channels / node.group, node.kernel_spatial[0], node.kernel_spatial[1]]
    weights.shape = int64_array(shape)
    weights.value.shape = weights.shape

    mark_input_bins(node)
    assign_dims_to_weights(weights, [2, 3], [1], [0], 4)


def tf_conv2d_infer(node, is_depthwise_conv):
    """
    The main purpose of this function to infer real pad values from TF padding scheme, e.g. 'SAME'.
    As a side effect it also delivers output shape size in spatial dimensions.
    It is stored and later validated/specialized by other dimensions in a generic conv infer function.
    TODO Now all inference is implemented here, no generic conv
    """
    input_shape = node.in_node(0).shape
    kernel_shape = node.in_node(1).shape

    if input_shape is None or kernel_shape is None or node.spatial_dims is None or node.stride is None:
        return
    spatial_dims = node.spatial_dims
    input_spatial_shape = np.array(input_shape[spatial_dims])
    stride_spatial_shape = np.array(node.stride[spatial_dims])
    kernel_spatial = np.array(kernel_shape[0:len(spatial_dims)])  # kernel spatial dims go first
    node.kernel_spatial = np.array(kernel_spatial)
    node.pad_spatial_shape, node.output_spatial_shape = tf_window_op_pad_infer(input_spatial_shape, kernel_spatial,
                                                                               stride_spatial_shape, node.auto_pad)

    # This is where a generic part starts

    pad = np.zeros((len(input_shape), 2), dtype=np.int64)
    pad[spatial_dims] = node.pad_spatial_shape

    output_shape = np.full_like(input_shape, -1, dtype=np.int64)
    output_shape[node.batch_dims] = input_shape[node.batch_dims]
    assert (len(node.channel_dims) == 1)
    assert (np.all(input_shape[node.channel_dims] == kernel_shape[-2]))
    if is_depthwise_conv:
        output_shape[node.channel_dims] = kernel_shape[-1] * kernel_shape[-2]
        node.group = kernel_shape[-2]
    else:
        output_shape[node.channel_dims] = kernel_shape[-1]
        node.group = 1
    output_shape[spatial_dims] = node.output_spatial_shape

    node.pad = pad
    node.output_shape = output_shape
    node.out_node().shape = output_shape

    mark_input_bins(node)
    assign_dims_to_weights(node.in_node(1), [0, 1], node.input_feature_channel, node.output_feature_channel, 4)
