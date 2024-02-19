# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.lrn import AttributedLRN


class LRNExtractor(FrontExtractorOp):
    op = 'LRN'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        alpha = attrs.float("alpha", 0.0001)
        beta = attrs.float("beta", 0.75)
        knorm = attrs.float("knorm", 2.0)
        nsize = attrs.int("nsize", None)

        AttributedLRN.update_node_stat(node, {
            'alpha': alpha,
            'beta': beta,
            'bias': knorm,
            'local_size': nsize,
        })
        return cls.enabled
