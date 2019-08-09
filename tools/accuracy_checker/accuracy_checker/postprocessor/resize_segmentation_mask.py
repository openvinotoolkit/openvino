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
from functools import singledispatch
import scipy.misc
import numpy as np

from ..config import NumberField
from ..utils import get_size_from_config
from .postprocessor import PostprocessorWithSpecificTargets, PostprocessorWithTargetsConfigValidator
from ..representation import SegmentationPrediction, SegmentationAnnotation


class ResizeMaskConfigValidator(PostprocessorWithTargetsConfigValidator):
    size = NumberField(floats=False, optional=True, min_value=1)
    dst_width = NumberField(floats=False, optional=True, min_value=1)
    dst_height = NumberField(floats=False, optional=True, min_value=1)

class ResizeSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'resize_segmentation_mask'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )
    _config_validator_type = ResizeMaskConfigValidator

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotation, prediction):
        target_height = self.dst_height or self.image_size[0]
        target_width = self.dst_width or self.image_size[1]

        @singledispatch
        def resize_segmentation_mask(entry, height, width):
            return entry

        @resize_segmentation_mask.register(SegmentationPrediction)
        def _(entry, height, width):
            entry_mask = []
            for class_mask in entry.mask:
                resized_mask = scipy.misc.imresize(class_mask, (height, width), 'nearest')
                entry_mask.append(resized_mask)
            entry.mask = np.array(entry_mask)

            return entry

        @resize_segmentation_mask.register(SegmentationAnnotation)
        def _(entry, height, width):
            entry.mask = scipy.misc.imresize(entry.mask, (height, width), 'nearest')
            return entry

        for target in annotation:
            resize_segmentation_mask(target, target_height, target_width)

        for target in prediction:
            resize_segmentation_mask(target, target_height, target_width)

        return annotation, prediction
