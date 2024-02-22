# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.MatMul import FullyConnected
from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import collect_until_token
from openvino.tools.mo.front.kaldi.utils import read_binary_matrix


class LinearComponentFrontExtractor(FrontExtractorOp):
    op = 'linearcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        collect_until_token(pb, b'<Params>')
        weights, weights_shape = read_binary_matrix(pb)
        
        mapping_rule = {
            'out-size': weights_shape[0],
            'transpose_weights': True,
        }

        embed_input(mapping_rule, 1, 'weights', weights)

        FullyConnected.update_node_stat(node, mapping_rule)
        return cls.enabled
