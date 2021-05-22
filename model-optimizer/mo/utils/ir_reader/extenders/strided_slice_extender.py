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
            # We can not use op.has_and_set(attr) here as a condition, because it will return False if begin/end is
            # 1D tensor and begin_mask/end_mask is equal to 0
            if op.has(attr) and op[attr] != '':
                Extender.attr_to_list(op, attr)
            else:
                assert attr not in ['begin_mask', 'end_mask'],\
                    '{} is not defined for the node {}'.format(attr, op.soft_get('name', op.id))
                op[attr] = int64_array([0])

        op.begin_mask = int64_array([1 - i for i in op.begin_mask])
        op.end_mask = int64_array([1 - i for i in op.end_mask])
