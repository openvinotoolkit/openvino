# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.identity import Identity
from mo.front.extractor import FrontExtractorOp


class ClipGradientComponentFrontExtractor(FrontExtractorOp):
    op = 'clipgradientcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        Identity.update_node_stat(node, {})
        return cls.enabled
