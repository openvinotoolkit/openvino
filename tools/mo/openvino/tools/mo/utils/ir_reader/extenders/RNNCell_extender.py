# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class RNNCell_extender(Extender):
    op = 'RNNCell'

    @staticmethod
    def extend(op: Node):
        if not op.has_valid('activations'):
            op['activations'] = None
