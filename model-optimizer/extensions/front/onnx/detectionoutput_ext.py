"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from math import log
import numpy as np

from extensions.ops.detectionoutput_onnx import ExperimentalDetectronDetectionOutput
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronDetectionOutputFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronDetectionOutput'
    enabled = True

    @staticmethod
    def extract(node):
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
        return __class__.enabled
