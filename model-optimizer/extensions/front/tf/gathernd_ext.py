# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.gathernd import GatherND
from mo.front.extractor import FrontExtractorOp


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
