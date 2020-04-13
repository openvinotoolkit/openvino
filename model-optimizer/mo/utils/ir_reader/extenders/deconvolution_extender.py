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


class ConvolutionBackpropData_extender(Extender):
    op = 'ConvolutionBackpropData'

    @staticmethod
    def extend(op: Node):
        common_backpropdata_extender(op)


class GroupConvolutionBackpropData_extender(Extender):
    op = 'GroupConvolutionBackpropData'

    @staticmethod
    def extend(op: Node):
        common_backpropdata_extender(op)


def common_backpropdata_extender(op: Node):
    if op.has_valid('output_padding'):
        op.output_padding = int64_array([0, 0] + op.output_padding)

    dim = len(op.strides)

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

    op['infer'] = backpropdata_infer


def backpropdata_infer(op: Node):
    op['new_input_shapes'] = list()
    for n in op.in_nodes():
        op.new_input_shapes.append(op.in_node(n).shape)
    assert len(op.new_input_shapes) == len(op.old_input_shapes)

    for i in range(len(op.new_input_shapes)):
        assert np.array_equal(op.new_input_shapes[i], op.old_input_shapes[i]), 'Something wrong happened while ' \
                                                    '{} shape infer with type {}!'.format(op.name, op.type)

    Extender.const_shape_infer(op)
