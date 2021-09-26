# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.MatMul import FullyConnected
from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.utils import read_binary_matrix, read_binary_vector, read_learning_info


class AffineTransformFrontExtractor(FrontExtractorOp):
    op = 'affinetransform'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        read_learning_info(pb)
        weights, weights_shape = read_binary_matrix(pb)
        biases = read_binary_vector(pb)

        mapping_rule = {
            'out-size': weights_shape[0],
            'transpose_weights': True,
        }
        embed_input(mapping_rule, 1, 'weights', weights)
        embed_input(mapping_rule, 2, 'biases', biases)

        FullyConnected.update_node_stat(node, mapping_rule)
        return cls.enabled
