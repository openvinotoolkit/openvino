# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class VariableExtractor(FrontExtractorOp):
    op = 'Variable'
    enabled = True

    @classmethod
    def extract(cls, node):
        Op.update_node_stat(node, {'op': 'FakeConst'})
        return cls.enabled


class VariableV2Extractor(FrontExtractorOp):
    op = 'VariableV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        Op.update_node_stat(node, {'op': 'FakeConst'})
        return cls.enabled
