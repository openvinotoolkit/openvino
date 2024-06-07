# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender
from openvino.tools.mo.utils.ir_reader.extenders.conv_extender import Conv_extender


class DeformableConv_extender(Extender):
    op = 'DeformableConvolution'

    @staticmethod
    def extend(op: Node):
        Conv_extender.extend(op)
        op['bias_addable'] = False,
        op['bias_term'] = False,
        op['weights_index'] = 2
