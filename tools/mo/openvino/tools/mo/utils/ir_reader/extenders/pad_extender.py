# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class Pad_extender(Extender):
    op = 'Pad'

    @staticmethod
    def extend(op: Node):
        op['mode'] = op['pad_mode']
        del op['pad_mode']
