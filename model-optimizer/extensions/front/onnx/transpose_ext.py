# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class TransposeFrontExtractor(FrontExtractorOp):
    op = 'Transpose'
    enabled = True

    @classmethod
    def extract(cls, node):
        # In case of undefined 'perm' attribute, Transpose operation in ONNX reverse the dimensions
        order = onnx_attr(node, 'perm', 'ints', default=None)
        attrs = {
            'order': int64_array(order) if order is not None else None,
            'reverse_order': order is None
        }
        Transpose.update_node_stat(node, attrs)
        return cls.enabled