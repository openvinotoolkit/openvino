# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.front.extractor import MXNetCustomFrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class RPNProposalMXNetFrontExtractor(MXNetCustomFrontExtractorOp):
    op = 'proposal'
    enabled = True

    def extract(self, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        feat_stride = attrs.int("feat_stride", 16)
        ratio = attrs.tuple("ratios", float, (0.5, 1, 2))
        scale = attrs.tuple("scales", int, (4, 8, 16, 32))
        min_size = attrs.int("rpn_min_size", 16)
        pre_nms_topn = attrs.int("rpn_pre_nms_top_n", 6000)
        post_nms_topn = attrs.int("rpn_post_nms_top_n", 300)
        nms_thresh = attrs.float("threshold", 0.7)

        node_attrs = {
            'feat_stride': feat_stride,
            'base_size': 0,
            'min_size': min_size,
            'ratio': mo_array(ratio),
            'scale': mo_array(scale),
            'pre_nms_topn': pre_nms_topn,
            'post_nms_topn': post_nms_topn,
            'nms_thresh': nms_thresh,
        }

        ProposalOp.update_node_stat(node, node_attrs)
        return (True, node_attrs)
