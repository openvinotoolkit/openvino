# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.utils import read_binary_vector, read_learning_info
from mo.ops.scale_shift import ScaleShiftOp


class RescaleFrontExtractor(FrontExtractorOp):
    op = 'rescale'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        read_learning_info(pb)
        weights = read_binary_vector(pb)
        mapping_rule = {}
        embed_input(mapping_rule, 1, 'weights', weights)
        ScaleShiftOp.update_node_stat(node, mapping_rule)
        return cls.enabled
