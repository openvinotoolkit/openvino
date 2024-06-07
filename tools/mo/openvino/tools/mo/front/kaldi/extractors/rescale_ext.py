# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.utils import read_binary_vector, read_learning_info
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp


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
