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

from ..config import BoolField, NumberField
from ..representation import DetectionPrediction, DetectionAnnotation
from .postprocessor import PostprocessorWithSpecificTargets, PostprocessorWithTargetsConfigValidator


class ClipBoxes(PostprocessorWithSpecificTargets):
    __provider__ = 'clip_boxes'

    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )

    def validate_config(self):
        class _ClipConfigValidator(PostprocessorWithTargetsConfigValidator):
            dst_width = NumberField(floats=False, optional=True, min_value=1)
            dst_height = NumberField(floats=False, optional=True, min_value=1)
            size = NumberField(floats=False, optional=True, min_value=1)
            boxes_normalized = BoolField(optional=True)

        clip_config_validator = _ClipConfigValidator(
            self.__provider__, on_extra_argument=_ClipConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        clip_config_validator.validate(self.config)

    def configure(self):
        size = self.config.get('size')
        self.dst_height = size or self.config.get('dst_height')
        self.dst_width = size or self.config.get('dst_width')

        self.boxes_normalized = self.config.get('boxes_normalized', False)

    def process_image(self, annotation, prediction):
        target_height = self.dst_height or self.image_size[0]
        target_width = self.dst_width or self.image_size[1]

        max_width = target_width if not self.boxes_normalized else 1
        max_height = target_height if not self.boxes_normalized else 1

        for target in annotation:
            self._clip_boxes(target, (0, max_width), (0, max_height))
        for target in prediction:
            self._clip_boxes(target, (0, max_width), (0, max_height))

        return annotation, prediction

    @staticmethod
    def _clip_boxes(entry, width_range, height_range):
        entry.x_mins = entry.x_mins.clip(width_range[0], width_range[1])
        entry.x_maxs = entry.x_maxs.clip(width_range[0], width_range[1])
        entry.y_mins = entry.y_mins.clip(height_range[0], height_range[1])
        entry.y_maxs = entry.y_maxs.clip(height_range[0], height_range[1])

        return entry
