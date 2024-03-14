# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from openvino.tools.mo.utils.ir_reader.extender import Extender


class AvgPool_extender(Extender):
    op = 'AvgPool'

    @staticmethod
    def extend(op: Node):
        common_pool_extender(op)

        if 'exclude-pad' in op:
            op['exclude_pad'] = op['exclude-pad']
            del op['exclude-pad']


class MaxPool_extender(Extender):
    op = 'MaxPool'

    @staticmethod
    def extend(op: Node):
        common_pool_extender(op)


def common_pool_extender(op: Node):
    for attr in ['strides', 'pads_begin', 'pads_end', 'kernel', 'dilations']:
        Extender.attr_to_list(op, attr)
    op['stride'] = int64_array([1, 1] + op.strides)
    op['window'] = int64_array([1, 1] + op.kernel)
    op['kernel_spatial'] = op.kernel
    op['output_spatial_shape'] = None

    if op.has_valid('dilations'):
        op['dilation'] = int64_array([1, 1] + op.dilations)
    if op.has_valid('index_element_type'):
        op['index_element_type'] = destination_type_to_np_data_type(op.index_element_type)

    op['batch_dims'] = int64_array([0]),
    op['channel_dims'] = int64_array([1]),

    op['pool_method'] = 'max' if op.type == 'MaxPool' else 'avg'

    dim = len(op.pads_begin)

    assert dim in (1, 2, 3), '{}D {} not supported! Node name: {}'.format(dim, op.soft_get('type'), op.soft_get('name', op.id))

    pad = [[0, 0], [0, 0]]
    pad.extend([[op.pads_begin[i], op.pads_end[i]] for i in range(dim)])

    op['pad'] = int64_array(pad)

    op['spatial_dims'] = [i + 2 for i in range(dim)]

    if op.has_valid('rounding_type') and op.rounding_type == 'ceil':
        op['pooling_convention'] = 'full'
