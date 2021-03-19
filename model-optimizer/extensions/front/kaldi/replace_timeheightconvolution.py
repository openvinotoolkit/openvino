"""
 Copyright (C) 2021 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.convolution import Convolution
from mo.ops.memoryoffset import MemoryOffset


class ReplaceTimeHeightConvolutionPattern(FrontReplacementOp):
    op = "TimeHeightConvolution"
    enabled = True

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_before(self):
        from extensions.front.kaldi.add_permute_after_convolution import ReplaceConvolutionTranspose
        from extensions.front.kaldi.add_reshape_around_convolution import ReplaceConvolutionReshape
        from extensions.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
        from extensions.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [MemoryOffsetAdjustment, ReplaceConvolutionTranspose,
                ReplaceConvolutionReshape, SplitRecurrentMemoryOffset]

    def replace_op(self, graph: Graph, node: Node):
        # create memoryoffsets for context
        time_offsets = node.soft_get('time_offsets')
        if len(time_offsets) > 1:
            concat = Concat.create_node(attrs={'in_port_count': len(time_offsets)})
            i = 0
            for t in time_offsets:
                memoff = MemoryOffset.create_node(attrs={'t': t})
                concat.in_port(i).connect(memoff.out_port(0))
                i = i + 1
            prev = concat.out_port(0)
        elif len(time_offsets) == 1:
            memoff = MemoryOffset.create_node(attrs={'t': time_offsets[0]})
            prev = memoff.out_port(0)
        else:
            prev = node.in_port(0).get_source()

        patch_stride = len(time_offsets)
        stride = node.soft_get("height_subsample", 1)
        offsets = node.soft_get("offsets", [[]])
        kernel = [0, 0]
        kernel[0] = max(offsets[:][0]) - min(offsets[:][0]) + 1
        kernel[1] = max(offsets[:][1]) - min(offsets[:][1]) + 1
        if (patch_stride - kernel) % stride != 0:
            raise Error(
                'Kernel size and stride does not correspond to `patch_stride` attribute of Convolution layer. ' +
                refer_to_faq_msg(93))

        output = node.in_port(1).shape[1]

        mapping_rule = {
            'output': output,
            'patch_stride': patch_stride,
            'bias_term': None,
            'pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64),
            'pad_spatial_shape': np.array([[0, 0], [0, 0]], dtype=np.int64),
            'dilation': np.array([1, 1, 1, 1], dtype=np.int64),
            'kernel': np.array([1, 1, kernel[0], kernel[1]], dtype=np.int64),
            'stride': np.array([1, 1, 1, stride], dtype=np.int64),
            'kernel_spatial': np.array(kernel, dtype=np.int64),
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': [2, 3],
            'group': 1,
            'reshape_kernel': True,
            'bias_addable': True,
        }
        conv = Convolution.create_node(attrs=mapping_rule)
        conv.in_port(0).connect(prev)
        conv.in_port(1).connect(node.in_port(1).get_source())
        conv.in_port(2).connect(node.in_port(2).get_source())

        return [conv.id]
