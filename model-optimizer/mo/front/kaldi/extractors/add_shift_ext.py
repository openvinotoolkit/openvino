# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.utils import read_binary_vector, read_learning_info
from mo.ops.scale_shift import ScaleShiftOp


class AddShiftFrontExtractor(FrontExtractorOp):
    op = 'addshift'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        read_learning_info(pb)
        biases = read_binary_vector(pb)
        bias_term = True
        mapping_rule = {'bias_term': bias_term}
        embed_input(mapping_rule, 1, 'weights', np.ones(biases.shape))
        embed_input(mapping_rule, 2, 'biases', biases)
        ScaleShiftOp.update_node_stat(node, mapping_rule)
        return cls.enabled
