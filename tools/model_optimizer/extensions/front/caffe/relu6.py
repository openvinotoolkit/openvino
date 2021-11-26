# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.activation_ops import ReLU6
from mo.front.extractor import FrontExtractorOp


class ReLU6FrontExtractor(FrontExtractorOp):
    op = 'ReLU6'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU6.update_node_stat(node)
        return True
