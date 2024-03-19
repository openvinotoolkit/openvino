# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.scatternd import TFScatterND
from openvino.tools.mo.front.extractor import FrontExtractorOp


class ScatterNDExtractor(FrontExtractorOp):
    op = 'ScatterNd'
    enabled = True

    @classmethod
    def extract(cls, node):
        TFScatterND.update_node_stat(node, {})
        return cls.enabled
