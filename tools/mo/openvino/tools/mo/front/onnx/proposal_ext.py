# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.proposal_onnx import GenerateProposalsSingleImage
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class GenerateProposalsSingleImageFrontExtractor(FrontExtractorOp):
    op = 'GenerateProposals'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = dict(min_size=onnx_attr(node, 'min_size', 'f', 16.0),
                     nms_thresh=onnx_attr(node, 'nms_thresh', 'f', 0.7),
                     post_nms_topN=onnx_attr(node, 'post_nms_topN', 'i', 300),
                     pre_nms_topN=onnx_attr(node, 'pre_nms_topN', 'i', 6000),
                     spatial_scale=onnx_attr(node, 'legacy_plus_one', 'f', 1.0/16),
                     legacy_plus_one=onnx_attr(node, 'legacy_plus_one', 'b', True)
                     )
        GenerateProposalsSingleImage.update_node_stat(node, attrs)
        return cls.enabled
