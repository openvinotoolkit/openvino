# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.split import AttributedVariadicSplit, AttributedSplit
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, onnx_get_num_outputs


class SplitFrontExtractor(FrontExtractorOp):
    op = 'Split'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', default=0, dst_type=np.int64)
        size_splits = onnx_attr(node, 'split', 'ints', default=None, dst_type=int64_array)
        if size_splits is None:
            AttributedSplit.update_node_stat(node, {
                'axis': axis,
                'num_splits': onnx_get_num_outputs(node),
            })
        else:
            AttributedVariadicSplit.update_node_stat(node, {
                'axis': axis,
                'size_splits': size_splits,
            })
        return cls.enabled
