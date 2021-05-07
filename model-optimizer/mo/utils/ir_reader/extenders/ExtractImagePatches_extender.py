# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class ExtractImagePatches(Extender):
    op = 'ExtractImagePatches'

    @staticmethod
    def extend(op: Node):
        op['sizes'] = int64_array([1, 1] + op.sizes)
        op['strides'] = int64_array([1, 1] + op.strides)
        op['rates'] = int64_array([1, 1] + op.rates)

        op['spatial_dims'] = int64_array([2, 3])
