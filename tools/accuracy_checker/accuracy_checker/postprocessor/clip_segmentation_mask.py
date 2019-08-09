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
from .postprocessor import PostprocessorWithSpecificTargets, PostprocessorWithTargetsConfigValidator
from ..representation import BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction
from ..config import NumberField, ConfigError


class ClipMaskConfigValidator(PostprocessorWithTargetsConfigValidator):
    min_value = NumberField(floats=False, min_value=0, optional=True)
    max_value = NumberField(floats=False)


class ClipSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'clip_segmentation_mask'

    annotation_types = (BrainTumorSegmentationAnnotation, )
    prediction_types = (BrainTumorSegmentationPrediction, )
    _config_validator_type = ClipMaskConfigValidator

    def configure(self):
        self.min_value = self.config.get('min_value', 0)
        self.max_value = self.config['max_value']
        if self.max_value < self.min_value:
            raise ConfigError('max_value should be greater than min_value')

    def process_image(self, annotation, prediction):
        for target in annotation:
            target.mask = np.clip(target.mask, a_min=self.min_value, a_max=self.max_value)

        for target in prediction:
            target.mask = np.clip(target.mask, a_min=self.min_value, a_max=self.max_value)

        return annotation, prediction
