# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class ExtractImagePatches(Extender):
    op = 'ExtractImagePatches'

    @staticmethod
    def extend(op: Node):
        op['sizes'] = int64_array([1, 1] + op.sizes)
        op['strides'] = int64_array([1, 1] + op.strides)
        op['rates'] = int64_array([1, 1] + op.rates)

        op['spatial_dims'] = int64_array([2, 3])
