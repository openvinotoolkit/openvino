# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.argmin import ArgMinOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class ArgMinFrontExtractor(FrontExtractorOp):
    op = 'ArgMin'
    enabled = True

    @classmethod
    def extract(cls, node):
        keepdims = onnx_attr(node, 'keepdims', 'i', default=1)
        axis = onnx_attr(node, 'axis', 'i', default=0)

        attrs = {
            'axis': axis,
            'top_k': 1,
            'keepdims': keepdims,
            'remove_values_output': True
        }

        ArgMinOp.update_node_stat(node, attrs)
        return cls.enabled
