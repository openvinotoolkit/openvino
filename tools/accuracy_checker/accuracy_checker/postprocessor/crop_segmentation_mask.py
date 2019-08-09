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

from .postprocessor import PostprocessorWithSpecificTargets, PostprocessorWithTargetsConfigValidator
from ..representation import BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction
from ..config import NumberField
from ..preprocessor import Crop3D
from ..utils import get_size_3d_from_config


class CropMaskConfigValidator(PostprocessorWithTargetsConfigValidator):
    size = NumberField(floats=False, min_value=1)
    dst_width = NumberField(floats=False, optional=True, min_value=1)
    dst_height = NumberField(floats=False, optional=True, min_value=1)
    dst_volume = NumberField(floats=False, optional=True, min_value=1)


class CropSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'crop_segmentation_mask'

    annotation_types = (BrainTumorSegmentationAnnotation,)
    prediction_types = (BrainTumorSegmentationPrediction,)
    _config_validator_type = CropMaskConfigValidator

    def configure(self):
        self.dst_height, self.dst_width, self.dst_volume = get_size_3d_from_config(self.config)

    def process_image(self, annotation, prediction):
        for target in annotation:
            target.mask = Crop3D.crop_center(target.mask, self.dst_height, self.dst_width, self.dst_volume)

        for target in prediction:
            target.mask = Crop3D.crop_center(target.mask, self.dst_height, self.dst_width, self.dst_volume)

        return annotation, prediction
