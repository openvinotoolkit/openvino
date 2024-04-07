# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.ops.gather import AttributedGather
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class GatherFrontExtractor(FrontExtractorOp):
    op = 'Gather'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'axis': int64_array(onnx_attr(node, 'axis', 'i', default=0))
        }

        AttributedGather.update_node_stat(node, attrs)
        return cls.enabled
