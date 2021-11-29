# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import collect_until_token
from mo.front.kaldi.utils import read_binary_matrix
from mo.ops.lstmnonlinearity import LstmNonLinearity


class LSTMNonlinearityFrontExtractor(FrontExtractorOp):
    op = 'lstmnonlinearitycomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        collect_until_token(pb, b'<Params>')
        ifo_x_weights, ifo_x_weights_shape = read_binary_matrix(pb)

        mapping_rule = {}

        assert len(ifo_x_weights_shape) == 2, "Unexpected shape of weights in LSTMNonLinearityComponent"
        assert ifo_x_weights_shape[0] == 3, "Unexpected shape of weights in LSTMNonLinearityComponent"

        ifo_x_weights = ifo_x_weights.reshape(ifo_x_weights_shape)
        embed_input(mapping_rule, 1, 'i_weights', ifo_x_weights[0][:])
        embed_input(mapping_rule, 2, 'f_weights', ifo_x_weights[1][:])
        embed_input(mapping_rule, 3, 'o_weights', ifo_x_weights[2][:])

        LstmNonLinearity.update_node_stat(node, mapping_rule)
        return cls.enabled
