# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.gatherelements import GatherElements
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class GatherElementsFrontExtractor(FrontExtractorOp):
    op = 'GatherElements'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'axis': onnx_attr(node, 'axis', 'i', default=0)
        }
        GatherElements.update_node_stat(node, attrs)
        return cls.enabled
