# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class ProposalFrontExtractor(FrontExtractorOp):
    op = '_contrib_Proposal'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        pre_nms_topn = attrs.int('rpn_pre_nms_top_n', 6000)
        post_nms_topn = attrs.int('rpn_post_nms_top_n', 300)
        nms_thresh = attrs.float('threshold', 0.7)
        min_size = attrs.int('rpn_min_size', 16)
        scale = attrs.tuple("scales", float, (4, 8, 16, 32))
        ratio = attrs.tuple("ratios", float, (0.5, 1, 2))
        feat_stride = attrs.int('feature_stride', 16)

        update_attrs = {
            'feat_stride': feat_stride,
            'ratio': mo_array(ratio),
            'min_size': min_size,
            'scale': mo_array(scale),
            'pre_nms_topn': pre_nms_topn,
            'post_nms_topn': post_nms_topn,
            'nms_thresh': nms_thresh,
            'base_size': feat_stride
        }

        # update the attributes of the node
        ProposalOp.update_node_stat(node, update_attrs)
        return cls.enabled
