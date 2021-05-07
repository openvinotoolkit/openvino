# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.activation_ops import Tanh
from mo.front.extractor import FrontExtractorOp


class TanhFrontExtractor(FrontExtractorOp):
    op = 'Tanh'
    enabled = True

    @classmethod
    def extract(cls, node):
        Tanh.update_node_stat(node)
        return cls.enabled

