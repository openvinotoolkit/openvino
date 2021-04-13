# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.convolution import Convolution
from mo.ops.memoryoffset import MemoryOffset


class ReplaceTimeHeightConvolutionPattern(FrontReplacementOp):
    op = "timeheightconvolutioncomponent"
    enabled = True

    def run_after(self):
        from extensions.front.MoveEmbeddedInputsToInputs import MoveEmbeddedInputsToInputs
        from extensions.front.restore_ports import RestorePorts
        return [MoveEmbeddedInputsToInputs, RestorePorts]

    def run_before(self):
        from extensions.front.kaldi.add_permute_after_convolution import ReplaceConvolutionTranspose
        from extensions.front.kaldi.add_reshape_around_convolution import ReplaceConvolutionReshape
        from extensions.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
        from extensions.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [MemoryOffsetAdjustment, ReplaceConvolutionReshape, ReplaceConvolutionTranspose,
                SplitRecurrentMemoryOffset]

    def replace_op(self, graph: Graph, node: Node):
        req_time_offsets = node.soft_get('time_offsets')
        offsets = node.soft_get("offsets", [[]])
        all_time_offsets = set(offsets[:, 0])
        in_name = node.soft_get('name', node.id)

        # input for convolution
        prev = node.in_port(0).get_source()
        # create memoryoffsets for context gathering
        # we need concat if time offsets more than 1
        if len(all_time_offsets) > 1:
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
            prev = concat.out_port(0)
        # no need to create concat for 1 offset
        elif len(all_time_offsets) == 1:
            t = all_time_offsets.pop()
            has_default = t not in req_time_offsets
            memoff = MemoryOffset(graph, attrs={'name': in_name + '/MemoryOffset',
                                                'has_default': has_default, 'splitted': False,
                                                'pair_name': in_name + '/MemoryOffset_pair',
                                                't': t}).create_node()
            memoff.in_port(0).connect(node.in_port(0).get_source())
            prev = memoff.out_port(0)

        stride = node.soft_get("height_subsample", 1)

        kernel = [0, 0]
        kernel[0] = len(set(offsets[:, 0]))
        kernel[1] = len(set(offsets[:, 1]))

        pad_h = [0, 0]
        pad_h[0] = -min(offsets[:, 1]) if min(offsets[:, 1]) < 0 else 0
        pad_h[1] = stride * node.height_out - (node.height_in - kernel[1] + 1 + pad_h[0])

        dilation_t = (max(offsets[:, 0]) - min(offsets[:, 0])) / (kernel[0] - 1)
        dilation_h = (max(offsets[:, 1]) - min(offsets[:, 1])) / (kernel[1] - 1)

        mapping_rule = {
            'name': in_name + '/Convolution',
            'output': node['out_channels'],
            'patch_stride': node.height_in * node.in_channels,
            'bias_term': None,
            'pad': np.array([[0, 0], [0, 0], [0, 0], pad_h], dtype=np.int64),
            'pad_spatial_shape': np.array([[0, 0], pad_h], dtype=np.int64),
            'dilation': np.array([1, 1, dilation_t, dilation_h], dtype=np.int64),
            'kernel': np.array([node.out_channels, node.in_channels, kernel[0], kernel[1]], dtype=np.int64),
            'stride': np.array([1, 1, 1, stride], dtype=np.int64),
            'kernel_spatial': np.array(kernel, dtype=np.int64),
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'channel_dims': int64_array([1]),
            'spatial_dims': int64_array([2, 3]),
            'batch_dims': int64_array([0]),
            'kernel_spatial_idx': [2, 3],
            'group': 1,
            'reshape_kernel': True,
            'bias_addable': True,
        }
        conv = Convolution(graph, attrs=mapping_rule).create_node()
        conv.in_port(0).connect(prev)
        conv.in_port(1).connect(node.in_port(1).get_source())

        # change layout for weights from OHWI to OIHW
        # in future should be replaced by common Permute mechanics
        weights = conv.in_port(1).get_source().node.value
        weights = weights.reshape(int64_array([node.out_channels, -1, node.in_channels]))
        weights = weights.transpose(int64_array([0, 2, 1]))
        weights = weights.flatten()
        conv.in_port(1).get_source().node.value = weights

        conv.in_port(2).connect(node.in_port(2).get_source())

        return [conv.id]
