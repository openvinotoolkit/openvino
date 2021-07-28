# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender
from mo.utils.ir_reader.extenders.conv_extender import Conv_extender


class BinaryConv_extender(Extender):
    op = 'BinaryConvolution'

    @staticmethod
    def extend(op: Node):
        Conv_extender.extend(op)
        op['type_to_create'] = 'Convolution'
