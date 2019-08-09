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
import math
import cv2

from .postprocessor import Postprocessor, BasePostprocessorConfig
from ..representation import SegmentationAnnotation, SegmentationPrediction
from ..config import NumberField, ConfigError


class ExtendSegmentationMaskConfigValidator(BasePostprocessorConfig):
    filling_label = NumberField(optional=True, floats=False)


class ExtendSegmentationMask(Postprocessor):
    """
    Extend annotation segmentation mask to prediction size filling border with specific label.
    """

    __provider__ = 'extend_segmentation_mask'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )
    _config_validator_type = ExtendSegmentationMaskConfigValidator

    def configure(self):
        self.filling_label = self.config.get('filling_label', 255)

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            annotation_mask = annotation_.mask
            dst_height, dst_width = prediction_.mask.shape[-2:]
            height, width = annotation_mask.shape[-2:]
            if dst_width < width or dst_height < height:
                raise ConfigError('size for extending should be not less current mask size')
            pad = []
            pad.append(int(math.floor((dst_height - height) / 2.0)))
            pad.append(int(math.floor((dst_width - width) / 2.0)))
            pad.append(int(dst_height - height - pad[0]))
            pad.append(int(dst_width - width - pad[1]))

            extended_mask = cv2.copyMakeBorder(
                annotation_mask, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=self.filling_label
            )
            annotation_.mask = extended_mask

        return annotation, prediction
