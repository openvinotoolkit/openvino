# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.lrn import AttributedLRN


class LRNExtractor(FrontExtractorOp):
    op = 'LRN'
    enabled = True

    @classmethod
    def extract(cls, node):
        param = node.pb.lrn_param
        region = 'same' if param.norm_region == 1 else 'across'

        AttributedLRN.update_node_stat(node, {
            'alpha': param.alpha,
            'beta': param.beta,
            'bias': param.k,
            'local_size': param.local_size,
            'region': region,
        })
        return cls.enabled
