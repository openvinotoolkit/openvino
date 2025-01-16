# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.argmax import ArgMaxOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr

class ArgMaxFrontExtractor(FrontExtractorOp):
    op = 'ArgMax'
    enabled = True

    @classmethod
    def extract(cls, node):
        keepdims = onnx_attr(node, 'keepdims', 'i', default=1)
        axis = onnx_attr(node, 'axis', 'i', default=0)

        attrs = {
            'axis': axis,

            # ONNX ArgMax always computes an index of one maximum value
            'top_k' : 1,
            'out_max_val' : 0,

            # Set attribute to trigger ArgMax replacer in case do not keep the dimension
            'keepdims': keepdims,

            'remove_values_output': True
        }

        ArgMaxOp.update_node_stat(node, attrs)
        return cls.enabled
