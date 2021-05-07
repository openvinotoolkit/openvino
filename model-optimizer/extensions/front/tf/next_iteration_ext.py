# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node


class NextIterationExtractor(FrontExtractorOp):
    op = "NextIteration"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        node['is_cyclic'] = True
        node['infer'] = copy_shape_infer
        return cls.enabled
