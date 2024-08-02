# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.identity import Identity
from openvino.tools.mo.front.extractor import FrontExtractorOp


class NoOpFrontExtractor(FrontExtractorOp):
    op = 'noopcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        Identity.update_node_stat(node)
        return cls.enabled
