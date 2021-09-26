# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.proposal import ProposalOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.extractor import FrontExtractorOp


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
            'ratio': np.array(param.ratio),
            'scale': np.array(param.scale),
            'pre_nms_topn': param.pre_nms_topn,
            'post_nms_topn': param.post_nms_topn,
            'nms_thresh': param.nms_thresh
        }

        mapping_rule = merge_attrs(param, update_attrs)
        # update the attributes of the node
        ProposalOp.update_node_stat(node, mapping_rule)
        return cls.enabled
