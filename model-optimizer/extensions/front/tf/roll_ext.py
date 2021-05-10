# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.roll import Roll
from mo.front.extractor import FrontExtractorOp


class RollExtractor(FrontExtractorOp):
    op = 'Roll'
    enabled = True

    @classmethod
    def extract(cls, node):
        Roll.update_node_stat(node, {})
        return cls.enabled
