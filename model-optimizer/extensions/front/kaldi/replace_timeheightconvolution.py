# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Node, Graph, rename_node
from mo.ops.concat import Concat
from mo.ops.convolution import Convolution
from mo.ops.memoryoffset import MemoryOffset


class ReplaceTimeHeightConvolutionPattern(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from extensions.front.MoveEmbeddedInputsToInputs import MoveEmbeddedInputsToInputs
        return [MoveEmbeddedInputsToInputs]

    def run_before(self):
        from extensions.front.kaldi.add_permute_after_convolution import ReplaceConvolutionTranspose
        from extensions.front.kaldi.add_reshape_around_convolution import ReplaceConvolutionReshape
        from extensions.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
        from extensions.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [MemoryOffsetAdjustment, ReplaceConvolutionReshape, ReplaceConvolutionTranspose,
                SplitRecurrentMemoryOffset]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='timeheightconvolutioncomponent'):
            self.replace_timeheightconv(graph, node)

    def replace_timeheightconv(self, graph: Graph, node: Node):
        req_time_offsets = node.soft_get('time_offsets')
        offsets = node.soft_get("offsets", [[]])
        all_time_offsets = list(set(offsets[:, 0]))
        all_time_offsets.sort()
        in_name = node.soft_get('name', node.id)
        rename_node(node, in_name + '/to_delete')

        # create memoryoffsets for context gathering
        # we need concat if time offsets more than 1
        concat = Concat(graph, attrs={'name': in_name + '/Concat',
                                      'in_ports_count': len(all_time_offsets)}).create_node()
        i = 0
        for t in all_time_offsets:
            # if time offset included in required_time_offsets we don't need default value
            has_default = t not in req_time_offsets
            memoff = MemoryOffset(graph, attrs={'name': in_name + '/MemoryOffset_' + str(i),
                                                't': t, 'has_default': has_default, 'splitted': False,
                                                'pair_name': in_name + '/MemoryOffset_pair_' + str(i)}).create_node()
            concat.in_port(i).connect(memoff.out_port(0))
            memoff.in_port(0).connect(node.in_port(0).get_source())
            i = i + 1

        stride = node.soft_get("height_subsample", 1)

        kernel = int64_array([0, 0])
        kernel[0] = len(set(offsets[:, 0]))
        kernel[1] = len(set(offsets[:, 1]))

        pad_h = int64_array([0, 0])
        pad_h[0] = -min(offsets[:, 1]) if min(offsets[:, 1]) < 0 else 0
        pad_h[1] = stride * node.height_out - (node.height_in - max([max(offsets[:, 1]), 0]))

        dilation_t = (max(offsets[:, 0]) - min(offsets[:, 0])) / (kernel[0] - 1) if kernel[0] > 1 else 1
        dilation_h = (max(offsets[:, 1]) - min(offsets[:, 1])) / (kernel[1] - 1) if kernel[0] > 1 else 1

        conv_attrs = {
            'name': in_name,
            'output': node['out_channels'],
            'height_in': node.height_in,
            'bias_term': None,
            'pad': int64_array([[0, 0], [0, 0], [0, 0], pad_h]),
            'pad_spatial_shape': int64_array([[0, 0], pad_h]),
            'dilation': int64_array([1, 1, dilation_t, dilation_h]),
            'kernel': int64_array([node.out_channels, node.in_channels, kernel[0], kernel[1]]),
            'stride': int64_array([1, 1, 1, stride]),
            'kernel_spatial': kernel,
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'channel_dims': int64_array([1]),
            'spatial_dims': int64_array([2, 3]),
            'batch_dims': int64_array([0]),
            'kernel_spatial_idx': int64_array([2, 3]),
            'group': 1,
            'reshape_kernel': True,
            'bias_addable': True,
        }
        conv = Convolution(graph, attrs=conv_attrs).create_node()
        conv.in_port(0).connect(concat.out_port(0))
        conv.in_port(1).connect(node.in_port(1).get_source())

        # change layout for weights from OHWI to OIHW
        # in future should be replaced by common Permute mechanics
        weights = conv.in_port(1).get_source().node.value
        weights = weights.reshape(int64_array([node.out_channels, -1, node.in_channels]))
        weights = weights.transpose(int64_array([0, 2, 1]))
        weights = weights.flatten()
        conv.in_port(1).get_source().node.value = weights

        conv.in_port(2).connect(node.in_port(2).get_source())
        node.out_port(0).get_connection().set_source(conv.out_port(0))
        graph.remove_node(node.id)
