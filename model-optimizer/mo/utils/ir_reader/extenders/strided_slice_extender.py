# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.ops.strided_slice import StridedSlice
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class StridedSlice_extender(Extender):
    op = 'StridedSlice'

    @staticmethod
    def extend(op: Node):
        for attr in StridedSlice.get_mask_names():
            Extender.attr_to_list(op, attr)

        op.begin_mask = int64_array([1 - i for i in op.begin_mask])
        op.end_mask = int64_array([1 - i for i in op.end_mask])
