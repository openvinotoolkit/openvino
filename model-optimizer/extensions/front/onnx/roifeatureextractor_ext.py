# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.roifeatureextractor_onnx import ExperimentalDetectronROIFeatureExtractor
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronROIFeatureExtractorFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronROIFeatureExtractor'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = dict(output_size=onnx_attr(node, 'output_size', 'i', 7),
                     sampling_ratio=onnx_attr(node, 'sampling_ratio', 'i', 2),
                     aligned=onnx_attr(node, 'aligned', 'i', 0),
                     num_classes=onnx_attr(node, 'num_classes', 'i', 81),
                     post_nms_count=onnx_attr(node, 'post_nms_count', 'i', 2000),
                     score_threshold=onnx_attr(node, 'score_threshold', 'f', 0.05),
                     pyramid_scales=np.array(onnx_attr(node, 'pyramid_scales', 'ints', [4, 8, 16, 32, 64]),
                                             dtype=np.int64),
                     )

        ExperimentalDetectronROIFeatureExtractor.update_node_stat(node, attrs)
        return cls.enabled
