# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class LSTMCell_extender(Extender):
    op = 'LSTMCell'

    @staticmethod
    def extend(op: Node):
        if not op.has_valid('activations'):
            op['activations'] = None
        op['infer'] = Extender.const_shape_infer
