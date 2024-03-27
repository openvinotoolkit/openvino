# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.topkrois_onnx import ExperimentalDetectronTopKROIs
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronTopKROIsFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronTopKROIs'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = dict(max_rois=onnx_attr(node, 'max_rois', 'i', 1000))
        ExperimentalDetectronTopKROIs.update_node_stat(node, attrs)
        return cls.enabled
