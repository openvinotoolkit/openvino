# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.topkrois_onnx import ExperimentalDetectronTopKROIs
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronTopKROIsFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronTopKROIs'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = dict(max_rois=onnx_attr(node, 'max_rois', 'i', 1000))
        ExperimentalDetectronTopKROIs.update_node_stat(node, attrs)
        return cls.enabled
