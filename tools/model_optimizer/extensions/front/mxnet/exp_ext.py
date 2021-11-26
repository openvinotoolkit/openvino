# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.activation_ops import Exp
from mo.front.extractor import FrontExtractorOp


class ExpExtractor(FrontExtractorOp):
    op = 'exp'
    enabled = True

    @classmethod
    def extract(cls, node):
        Exp.update_node_stat(node)
        return cls.enabled
