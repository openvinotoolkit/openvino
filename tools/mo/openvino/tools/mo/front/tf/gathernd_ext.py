# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.gathernd import GatherND
from openvino.tools.mo.front.extractor import FrontExtractorOp


class GatherNDFrontExtractor(FrontExtractorOp):
    op = 'GatherNd'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'batch_dims': 0,
        }
        GatherND.update_node_stat(node, attrs)
        return cls.enabled
