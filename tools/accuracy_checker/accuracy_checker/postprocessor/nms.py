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

from ..config import NumberField
from .postprocessor import BasePostprocessorConfig, Postprocessor
from ..representation import DetectionPrediction, DetectionAnnotation


class NMS(Postprocessor):
    __provider__ = 'nms'

    prediction_types = (DetectionPrediction, )
    annotation_types = (DetectionAnnotation, )

    def validate_config(self):
        class _NMSConfigValidator(BasePostprocessorConfig):
            overlap = NumberField(min_value=0, max_value=1, optional=True)

        nms_config_validator = _NMSConfigValidator(
            self.__provider__, on_extra_argument=_NMSConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        nms_config_validator.validate(self.config)

    def configure(self):
        self.overlap = self.config.get('overlap', 0.5)

    def process_image(self, annotations, predictions):
        for prediction in predictions:
            keep = self._nms(
                prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs, prediction.scores,
                self.overlap
            )
            prediction.remove([box for box in range(len(prediction.x_mins)) if box not in keep])

        return annotations, predictions

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, thresh):
        """
        Pure Python NMS baseline.
        """

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h

            union = (areas[i] + areas[order[1:]] - intersection)
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

            order = order[np.where(overlap <= thresh)[0] + 1]

        return keep
