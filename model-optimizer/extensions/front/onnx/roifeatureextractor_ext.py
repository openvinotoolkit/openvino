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

import numpy as np

from extensions.ops.roifeatureextractor_onnx import ExperimentalDetectronROIFeatureExtractor
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronROIFeatureExtractorFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronROIFeatureExtractor'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = dict(output_size=onnx_attr(node, 'output_size', 'i', 7),
                     sampling_ratio=onnx_attr(node, 'sampling_ratio', 'i', 2),
                     distribute_rois_between_levels=onnx_attr(node, 'distribute_rois_between_levels', 'i', 1),
                     preserve_rois_order=onnx_attr(node, 'preserve_rois_order', 'i', 1),
                     num_classes=onnx_attr(node, 'num_classes', 'i', 81),
                     post_nms_count=onnx_attr(node, 'post_nms_count', 'i', 2000),
                     score_threshold=onnx_attr(node, 'score_threshold', 'f', 0.05),
                     pyramid_scales=np.array(onnx_attr(node, 'pyramid_scales', 'ints', [4, 8, 16, 32, 64]),
                                             dtype=np.int64),
                     )

        ExperimentalDetectronROIFeatureExtractor.update_node_stat(node, attrs)
        return __class__.enabled
