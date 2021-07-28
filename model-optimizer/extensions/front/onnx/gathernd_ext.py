# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.gathernd import GatherND
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


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
