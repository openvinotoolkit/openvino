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

from ..config import BoolField, NumberField
from .postprocessor import BasePostprocessorConfig, Postprocessor
from ..representation import DetectionPrediction, DetectionAnnotation


class NMSConfigValidator(BasePostprocessorConfig):
    overlap = NumberField(min_value=0, max_value=1, optional=True)
    include_boundaries = BoolField(optional=True)
    keep_top_k = NumberField(min_value=0, optional=True)


class NMS(Postprocessor):
    __provider__ = 'nms'

    prediction_types = (DetectionPrediction, )
    annotation_types = (DetectionAnnotation, )
    _config_validator_type = NMSConfigValidator

    def configure(self):
        self.overlap = self.config.get('overlap', 0.5)
        self.include_boundaries = self.config.get('include_boundaries', True)
        self.keep_top_k = self.config.get('keep_top_k')

    def process_image(self, annotations, predictions):
        for prediction in predictions:
            keep = self.nms(
                prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs, prediction.scores,
                self.overlap, self.include_boundaries, self.keep_top_k
            )
            prediction.remove([box for box in range(len(prediction.x_mins)) if box not in keep])

        return annotations, predictions

    @staticmethod
    def nms(x1, y1, x2, y2, scores, thresh, include_boundaries=True, keep_top_k=None):
        """
        Pure Python NMS baseline.
        """

        b = 1 if include_boundaries else 0

        areas = (x2 - x1 + b) * (y2 - y1 + b)
        order = scores.argsort()[::-1]

        if keep_top_k:
            order = order[:keep_top_k]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + b)
            h = np.maximum(0.0, yy2 - yy1 + b)
            intersection = w * h

            union = (areas[i] + areas[order[1:]] - intersection)
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

            order = order[np.where(overlap <= thresh)[0] + 1]

        return keep
