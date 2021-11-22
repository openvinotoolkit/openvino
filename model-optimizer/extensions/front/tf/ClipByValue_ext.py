# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.ClipByValueTF import ClibByValueTF
from mo.front.extractor import FrontExtractorOp


class ClipByValueExtractor(FrontExtractorOp):
    op = 'ClipByValue'
    enabled = True

    @classmethod
    def extract(cls, node):
        ClibByValueTF.update_node_stat(node, {})
        return cls.enabled
