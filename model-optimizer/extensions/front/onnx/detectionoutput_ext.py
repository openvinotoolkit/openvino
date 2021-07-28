# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from math import log

import numpy as np

from extensions.ops.detectionoutput_onnx import ExperimentalDetectronDetectionOutput
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronDetectionOutputFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronDetectionOutput'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = dict(class_agnostic_box_regression=onnx_attr(node, 'class_agnostic_box_regression', 'i', 0),
                     max_detections_per_image=onnx_attr(node, 'max_detections_per_image', 'i', 100),
                     nms_threshold=onnx_attr(node, 'nms_threshold', 'f', 0.5),
                     num_classes=onnx_attr(node, 'num_classes', 'i', 81),
                     post_nms_count=onnx_attr(node, 'post_nms_count', 'i', 2000),
                     score_threshold=onnx_attr(node, 'score_threshold', 'f', 0.05),
                     max_delta_log_wh=onnx_attr(node, 'max_delta_log_wh', 'f', log(1000. / 16.)),
                     deltas_weights=np.array(onnx_attr(node, 'deltas_weights', 'floats', [10., 10., 5., 5.]),
                                             dtype=np.float32)
                     )
        ExperimentalDetectronDetectionOutput.update_node_stat(node, attrs)
        return cls.enabled
