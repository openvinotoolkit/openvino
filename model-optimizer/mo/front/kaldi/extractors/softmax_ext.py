# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.extractor import FrontExtractorOp
from mo.ops.softmax import Softmax


class SoftmaxComponentFrontExtractor(FrontExtractorOp):
    op = 'softmaxcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        return SoftmaxFrontExtractor.extract(node)


class SoftmaxFrontExtractor(FrontExtractorOp):
    op = 'softmax'
    enabled = True

    @classmethod
    def extract(cls, node):
        Softmax.update_node_stat(node, {'infer': copy_shape_infer})
        return cls.enabled
