# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


class ProposalFrontExtractor(FrontExtractorOp):
    op = 'Proposal'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.proposal_param
        update_attrs = {
            'feat_stride': param.feat_stride,
            'base_size': param.base_size,
            'min_size': param.min_size,
            'ratio': mo_array(param.ratio),
            'scale': mo_array(param.scale),
            'pre_nms_topn': param.pre_nms_topn,
            'post_nms_topn': param.post_nms_topn,
            'nms_thresh': param.nms_thresh
        }

        mapping_rule = merge_attrs(param, update_attrs)
        # update the attributes of the node
        ProposalOp.update_node_stat(node, mapping_rule)
        return cls.enabled
