# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


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
    for attr in ['strides', 'output_padding', 'pads_begin', 'pads_end', 'dilations']:
        Extender.attr_to_list(op, attr)

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
    Extender.use_shapes_from_ir(op)
