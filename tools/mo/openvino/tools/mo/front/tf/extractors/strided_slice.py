# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.strided_slice import StridedSlice


def int_to_array_bit_mask(im):
    list_repr = list(np.binary_repr(im))
    list_repr.reverse()
    list_repr = [int(li) for li in list_repr]
    return mo_array(list_repr, dtype=np.int32)


class StridedSliceFrontExtractor(FrontExtractorOp):
    op = 'StridedSlice'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.pb
        bm = int_to_array_bit_mask(pb.attr["begin_mask"].i)
        bm = mo_array([1 - b for b in bm], dtype=np.int32)
        em = int_to_array_bit_mask(pb.attr["end_mask"].i)
        em = mo_array([1 - b for b in em], dtype=np.int32)
        attrs = {
            'begin_mask': bm,
            'end_mask': em,
            'ellipsis_mask': int_to_array_bit_mask(pb.attr["ellipsis_mask"].i),
            'new_axis_mask': int_to_array_bit_mask(pb.attr["new_axis_mask"].i),
            'shrink_axis_mask': int_to_array_bit_mask(pb.attr["shrink_axis_mask"].i),
        }

        StridedSlice.update_node_stat(node, attrs)
        return cls.enabled
