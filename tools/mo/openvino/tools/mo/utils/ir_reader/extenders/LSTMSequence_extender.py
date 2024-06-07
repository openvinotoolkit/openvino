# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class LSTMSequence_extender(Extender):
    op = 'LSTMSequence'

    @staticmethod
    def extend(op: Node):
        op['infer'] = Extender.use_shapes_from_ir
