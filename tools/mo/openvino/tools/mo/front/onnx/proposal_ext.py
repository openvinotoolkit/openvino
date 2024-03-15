# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.proposal_onnx import ExperimentalDetectronGenerateProposalsSingleImage
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronGenerateProposalsSingleImageFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronGenerateProposalsSingleImage'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = dict(min_size=onnx_attr(node, 'min_size', 'f', 0.0),
                     nms_threshold=onnx_attr(node, 'nms_threshold', 'f', 0.7),
                     post_nms_count=onnx_attr(node, 'post_nms_count', 'i', 1000),
                     pre_nms_count=onnx_attr(node, 'pre_nms_count', 'i', 1000)
                     )
        ExperimentalDetectronGenerateProposalsSingleImage.update_node_stat(node, attrs)
        return cls.enabled
