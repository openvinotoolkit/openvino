# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.gathernd import GatherND
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class GatherNDFrontExtractor(FrontExtractorOp):
    op = 'GatherND'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'batch_dims': onnx_attr(node, 'batch_dims', 'i', default=0)
        }
        GatherND.update_node_stat(node, attrs)
        return cls.enabled
