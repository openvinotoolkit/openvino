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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.convolution import Convolution
from mo.ops.memoryoffset import MemoryOffset
from mo.ops.reshape import Reshape
from extensions.ops.transpose import Transpose



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
        # create memoryoffsets for context
        time_offsets = node.soft_get('time_offsets')
        in_name = node.soft_get('name', node.id)
        if len(time_offsets) > 1:
            concat = Concat(graph, attrs={'in_ports_count': len(time_offsets)}).create_node()
            i = 0
            for t in time_offsets:
                memoff = MemoryOffset(graph, attrs={'name': in_name +'/MemoryOffset',
                                                    't': t, 'has_default': False, 'splitted': False,
                                                    'pair_name': in_name + '/MemoryOffset_pair_' + str(t)}).create_node()
                concat.in_port(i).connect(memoff.out_port(0))
                memoff.in_port(0).connect(node.in_port(0).get_source())
                i = i + 1
            prev = concat.out_port(0)
        elif len(time_offsets) == 1:
            memoff = MemoryOffset(graph, attrs={'name': in_name +'/MemoryOffset',
                                                'has_default': False, 'splitted': False,
                                                'pair_name': in_name +'/MemoryOffset_pair',
                                                't': time_offsets[0]}).create_node()
            memoff.in_port(0).connect(node.in_port(0).get_source())
            prev = memoff.out_port(0)
        else:
            prev = node.in_port(0).get_source()

        #TODO: create subgraph to calculate shape
        #reshape = create_op_node_with_second_input(graph, Reshape,
        #                                           int64_array([1, len(time_offsets), node.height_in, node.in_channels]),
        #                                          {'name': 'Reshape_for_conv'}, prev.node)
        #transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 3, 1, 2]),
        #                                             {'name': node.name + '/Transpose'}, reshape)
        patch_stride = node.height_in * node.in_channels
        stride = node.soft_get("height_subsample", 1)
        offsets = node.soft_get("offsets", [[]])
        kernel = [0, 0]
        kernel[0] = max(offsets[:, 0]) - min(offsets[:, 0]) + 1
        kernel[1] = max(offsets[:, 1]) - min(offsets[:, 1]) + 1

        mapping_rule = {
            'output': node['out_channels'],
            'patch_stride': patch_stride,
            'bias_term': None,
            'pad': np.array([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.int64),
            'pad_spatial_shape': np.array([[0, 0], [1, 1]], dtype=np.int64),
            'dilation': np.array([1, 1, 1, 1], dtype=np.int64),
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
        weights = conv.in_port(1).get_source().node.value
        weights = weights.reshape(int64_array([node.out_channels, -1, node.in_channels]))
        weights = weights.transpose(int64_array([0, 2, 1]))
        weights = weights.flatten()
        conv.in_port(1).get_source().node.value = weights

        conv.in_port(2).connect(node.in_port(2).get_source())

        #reshape_out = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
        #                                               {'name': '/reshape_out'}, conv)

        return [conv.id]
