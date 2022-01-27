# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.eltwise import eltwise_infer
from openvino.tools.mo.front.common.partial_infer.utils import float32_array
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class Sqrt_extender(Extender):
    op = 'Sqrt'

    @staticmethod
    def operation(a):   # we use same function as for elementwise.power but with fixed
        if np.issubdtype(a.dtype, np.signedinteger):
            return float32_array(a.astype(np.float32) ** 0.5)
        return a ** 0.5

    @staticmethod
    def extend(op: Node):
        op['infer'] = lambda node: eltwise_infer(node, Sqrt_extender.operation)
