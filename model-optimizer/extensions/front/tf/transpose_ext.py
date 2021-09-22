# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.transpose import Transpose
from mo.front.extractor import FrontExtractorOp


class TransposeFrontExtractorTF(FrontExtractorOp):
    op = 'Transpose'
    enabled = True

    @classmethod
    def extract(cls, node):
        Transpose.update_node_stat(node, {'order': None})
        return cls.enabled
