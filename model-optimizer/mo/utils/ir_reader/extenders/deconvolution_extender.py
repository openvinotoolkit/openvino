"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.utils.ir_reader.extender import Extender
from mo.utils.graph import Node

from mo.front.common.partial_infer.utils import int64_array


class Deconv_extender(Extender):
    op = 'Deconvolution'

    # TODO Refactor all this extender, we have ConvolutionBackpropData/GroupConvolutionBackpropData instead of Deconvolution!

    @staticmethod
    def extend(op: Node):
        if op.has_valid('output_padding'):
            op.output_padding = int64_array([0, 0] + op.output_padding)

        dim = len(op.strides)

        # assert dim in (2, 3), '{}D Deconvolution not supported!'.format(dim)

        if op.has_valid('pads_begin') and op.has_valid('pads_end'):

            pad = [[0, 0], [0, 0]]
            pad.extend([[op.pads_begin[i], op.pads_end[i]] for i in range(dim)])

            op['pad'] = int64_array(pad)

        op['spatial_dims'] = [i + 2 for i in range(dim)]

        if not op.has_valid('dilations'):
            op['dilations'] = [1 for _ in range(dim)]
        if not op.has_valid('strides'):
            op['strides'] = [1 for _ in range(dim)]

        op['dilation'] = int64_array([1, 1] + op.dilations)
        op['stride'] = int64_array([1, 1] + op.strides)

        if not op.has_valid('old_type'):
            # op['batch_dims'] = int64_array([0])     # ?
            op['channel_dims'] = int64_array([1])

            op['input_feature_channel'] = 0
            op['output_feature_channel'] = 1

            op['kernel_spatial'] = op.kernel

            # if op.has_valid('auto_pad'):
            #     op['auto_pad'] = None

            op['infer'] = deconvolution_infer       # TODO Remove after supporting
        else:
            op['infer'] = backpropdata_infer


def deconvolution_infer(node: Node):
    dims = int64_array(node.in_node(0).shape)
    dilations = int64_array(node.dilations)
    strides = int64_array(node.strides)
    input_n = dims[0]
    kernel_shape = int64_array(node.kernel)
    kdims = np.where(dilations != 0, (kernel_shape - 1) * dilations + 1, kernel_shape)
    oc = node.output

    if node.has_valid('auto_pad') and node.auto_pad in ['valid', 'same_upper', 'same_lower']:
        auto_pad = node.auto_pad
        if auto_pad == 'valid':
            od_temp = (dims[2::] - 1) * strides + kdims
        else:
            od_temp = dims[2::] * strides
    else:
        od_temp = strides * (dims[2::] - 1) + kdims - node.pads_begin - node.pads_end

    out_shape = [input_n, oc]
    for d in od_temp:
        out_shape.append(np.int64(d))

    node['output_shape'] = int64_array(out_shape)
    for n in node.out_nodes():
        node.out_node(n).shape = node['output_shape']


def backpropdata_infer(op: Node):
    op['new_input_shapes'] = list()
    for n in op.in_nodes():
        op.new_input_shapes.append(op.in_node(n).shape)
    assert len(op.new_input_shapes) == len(op.old_input_shapes)

    for i in range(len(op.new_input_shapes)):
        assert np.array_equal(op.new_input_shapes[i], op.old_input_shapes[i]), 'Something wrong happened while ' \
                                                    '{} shape infer!'.format(op.old_type)

    output_shape = op.ports[len(op.in_nodes())]
    # op.output_shape = output_shape
    op.out_node().shape = int64_array(output_shape)
    op.type = op.old_type
