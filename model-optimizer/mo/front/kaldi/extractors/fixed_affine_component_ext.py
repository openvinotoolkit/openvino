# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.MatMul import FullyConnected
from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import find_next_tag, read_placeholder, collect_until_token
from mo.front.kaldi.utils import read_binary_matrix, read_binary_vector
from mo.utils.error import Error


class FixedAffineComponentFrontExtractor(FrontExtractorOp):
    op = 'fixedaffinecomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        collect_until_token(pb, b'<LinearParams>')
        weights, weights_shape = read_binary_matrix(pb)
        tag = find_next_tag(pb)
        read_placeholder(pb, 1)
        if tag != '<BiasParams>':
            raise Error('FixedAffineComponent must contain BiasParams')
        biases = read_binary_vector(pb)

        mapping_rule = {
            'out-size': weights_shape[0],
            'transpose_weights': True,
        }
        embed_input(mapping_rule, 1, 'weights', weights)
        embed_input(mapping_rule, 2, 'biases', biases)

        FullyConnected.update_node_stat(node, mapping_rule)
        return cls.enabled
