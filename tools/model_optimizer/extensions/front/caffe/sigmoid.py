# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.activation_ops import Sigmoid
from mo.front.extractor import FrontExtractorOp


class SigmoidFrontExtractor(FrontExtractorOp):
    op = 'Sigmoid'
    enabled = True

    @classmethod
    def extract(cls, node):
        Sigmoid.update_node_stat(node)
        return cls.enabled
