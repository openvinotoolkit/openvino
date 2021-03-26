# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.activation_ops import ReLU
from mo.front.extractor import FrontExtractorOp


class ReLUFrontExtractor(FrontExtractorOp):
    op = 'relu'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU.update_node_stat(node)
        return ReLUFrontExtractor.enabled
