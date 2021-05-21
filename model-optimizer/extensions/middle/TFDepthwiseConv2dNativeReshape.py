# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from extensions.middle.ConvertLayoutDependentOperations import ConvertLayoutDependentOperations
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class TFDepthwiseConv2dNativeReshape(MiddleReplacementPattern):
    """
    According to the specification of TF* DepthwiseConv2dNative kernel has shape
    [filter_height, filter_width, in_channels, channel_multiplier], and output_channels = in_channels * channel_multiplier.
    In general, we convert this operation to a GroupConvolution, by setting group = in_channels. So in IR kernel shape will be
    [in_channels, channel_multiplier, 1, filter_height, filter_width] <-> [G, O, I, H, W] (extensions/back/ConvolutionNormalizer.py)

    But there is a case when in_channels = 1 and it is converted to a Convolution, so we can not calculate the number of
    output_channels correctly. We will transpose weights from [filter_height, filter_width, in_channels, channel_multiplier]
    to [in_channels, channel_multiplier, filter_height, filter_width]. This is incorrect, because for Convolution op
    kernel shape should be [in_channels * channel_multiplier, 1, filter_height, filter_width] according to OpenVINO
    specification <-> [O, I, H, W].

    So we have to reshape kernel shape to [filter_height, filter_width, in_channels * channel_multiplier, 1] to transpose
    it correctly later
    """
    def run_before(self):
        return [ConvertLayoutDependentOperations]

    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op="DepthwiseConv2dNative"):
            node_name = node.soft_get('name', node.id)
            kernel_shape = node.in_port(1).data.get_shape()
            if kernel_shape[node.input_feature_channel] == 1:
                reshape_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, 0, -1, 1]),
                                                                op_attrs=dict(name=node_name + '/Reshape',
                                                                              override_output_shape=True))
                node.in_port(1).get_connection().insert_node(reshape_node)
