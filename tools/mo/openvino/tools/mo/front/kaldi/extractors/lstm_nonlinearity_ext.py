# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import collect_until_token, collect_until_token_and_read
from openvino.tools.mo.front.kaldi.utils import read_binary_matrix
from openvino.tools.mo.ops.lstmnonlinearity import LstmNonLinearity
from openvino.tools.mo.utils.error import Error


class LSTMNonlinearityFrontExtractor(FrontExtractorOp):
    op = 'lstmnonlinearitycomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        collect_until_token(pb, b'<Params>')
        ifo_x_weights, ifo_x_weights_shape = read_binary_matrix(pb)

        try:
            use_dropout = collect_until_token_and_read(pb, b'<UseDropout>', bool)
        except Error:
            # layer have not UseDropout attribute, so setup it to False
            use_dropout = False

        mapping_rule = {'use_dropout': use_dropout}

        assert len(ifo_x_weights_shape) == 2, "Unexpected shape of weights in LSTMNonLinearityComponent"
        assert ifo_x_weights_shape[0] == 3, "Unexpected shape of weights in LSTMNonLinearityComponent"

        ifo_x_weights = ifo_x_weights.reshape(ifo_x_weights_shape)
        embed_input(mapping_rule, 1, 'i_weights', ifo_x_weights[0][:])
        embed_input(mapping_rule, 2, 'f_weights', ifo_x_weights[1][:])
        embed_input(mapping_rule, 3, 'o_weights', ifo_x_weights[2][:])

        LstmNonLinearity.update_node_stat(node, mapping_rule)
        return cls.enabled
