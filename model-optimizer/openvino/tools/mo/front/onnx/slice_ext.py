# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import get_onnx_opset_version
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.slice import Slice, AttributedSlice
from mo.utils.error import Error


class SliceFrontExtractor(FrontExtractorOp):
    op = 'Slice'
    enabled = True

    @classmethod
    def extract(cls, node):
        if get_onnx_opset_version(node) < 10:
            starts = int64_array(onnx_attr(node, 'starts', 'ints', default=[]))
            ends = int64_array(onnx_attr(node, 'ends', 'ints', default=[]))
            axes = int64_array(onnx_attr(node, 'axes', 'ints', default=[]))

            if len(starts) == 0 or len(ends) == 0:
                raise Error("starts or/and ends are not specified for the node {}".format(node.name))
            if len(axes) == 0:
                axes = np.arange(len(starts), dtype=np.int)

            attrs = {'axes': axes, 'starts': starts, 'ends': ends}
            AttributedSlice.update_node_stat(node, attrs)
        else:  # onnx_opset_version >= 10
            Slice.update_node_stat(node)
        return cls.enabled
