"""
Copyright (C) 2018-2019 Intel Corporation

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

from .postprocessor import Postprocessor, BasePostprocessorConfig
from ..representation import SegmentationAnnotation, SegmentationPrediction
from ..config import NumberField


class ZoomSegMask(Postprocessor):
    """
    Zoom probabilities of segmentation prediction.
    """

    __provider__ = 'zoom_segmentation_mask'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    def validate_config(self):
        class _ZoomSegMaskConfigValidator(BasePostprocessorConfig):
            zoom = NumberField(floats=False, min_value=1)

        zoom_segmentation_mask_config_validator = _ZoomSegMaskConfigValidator(
            self.__provider__, on_extra_argument=_ZoomSegMaskConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        zoom_segmentation_mask_config_validator.validate(self.config)

    def configure(self):
        self.zoom = self.config['zoom']

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            height, width = annotation_.mask.shape[:2]
            prob = prediction_.mask
            zoom_prob = np.zeros((prob.shape[0], height, width), dtype=np.float32)
            for c in range(prob.shape[0]):
                for h in range(height):
                    for w in range(width):
                        r0 = h // self.zoom
                        r1 = r0 + 1
                        c0 = w // self.zoom
                        c1 = c0 + 1
                        rt = float(h) / self.zoom - r0
                        ct = float(w) / self.zoom - c0
                        v0 = rt * prob[c, r1, c0] + (1 - rt) * prob[c, r0, c0]
                        v1 = rt * prob[c, r1, c1] + (1 - rt) * prob[c, r0, c1]
                        zoom_prob[c, h, w] = (1 - ct) * v0 + ct * v1
            prediction_.mask = zoom_prob

        return annotation, prediction
