# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class Conv_extender(Extender):
    op = 'Convolution'

    @staticmethod
    def extend(op: Node):
        for attr in ['strides', 'dilations', 'pads_begin', 'pads_end', 'output_padding']:
            Extender.attr_to_list(op, attr)

        op['stride'] = int64_array([1, 1] + op.strides)
        op['dilation'] = int64_array([1, 1] + op.dilations)

        op['batch_dims'] = int64_array([0])
        op['channel_dims'] = int64_array([1])

        if op.has_valid('output_padding'):
            op.output_padding = int64_array([0, 0] + op.output_padding)

        # Be VERY careful with these attributes!
        op['input_feature_channel'] = 1
        op['output_feature_channel'] = 0

        dim = len(op.pads_begin)

        assert dim in (1, 2, 3), '{}D Convolution not supported!'.format(dim)

        pad = [[0, 0], [0, 0]]
        pad.extend([[op.pads_begin[i], op.pads_end[i]] for i in range(dim)])

        op['pad'] = int64_array(pad)

        op['spatial_dims'] = [i + 2 for i in range(dim)]
