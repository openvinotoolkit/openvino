# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.Reverse import Reverse
from mo.front.extractor import FrontExtractorOp


class ReverseV2FrontExtractor(FrontExtractorOp):
    op = 'ReverseV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        Reverse.update_node_stat(node)
        return cls.enabled
